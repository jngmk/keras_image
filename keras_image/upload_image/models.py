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


# @receiver(models.signals.pre_save, sender=Image)
# def auto_delete_file_on_change(sender, instance, **kwargs):
#     """
#     Deletes old file from filesystem
#     when corresponding `MediaFile` object is updated
#     with new file.
#     """
#     if not instance.pk:
#         return False

#     try:
#         old_file = MediaFile.objects.get(pk=instance.pk).file
#     except MediaFile.DoesNotExist:
#         return False

#     new_file = instance.file
#     if not old_file == new_file:
#         if os.path.isfile(old_file.path):
#             os.remove(old_file.path)


# class Image(models.Model):
#     image = models.ImageField(upload_to='images/')