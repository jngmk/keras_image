from django.db import models
from django.core.files.storage import FileSystemStorage
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill

# from django.dispatch import receiver

# fs = FileSystemStorage(location='/media/images')
# Create your models here.
class Image(models.Model):
    # image = models.ImageField(upload_to='images/')
    image = ProcessedImageField(
        upload_to='images/',
        processors=[ResizeToFill(90, 120)],
        format='JPEG',
    )
    update_date = models.DateTimeField(blank=True, null=False)

    class Meta:
        get_latest_by = 'update_date'

    def __str__(self):
        return str(self.image)