import os
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponseForbidden
from django import forms
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.views.generic import CreateView
from django.urls import reverse_lazy
from .models import Image
from .forms import ImageUploadForm
import numpy as numpy
from skimage import io
from skimage.transform import resize
import imageio
import dlib
import keras
import openface
import cv2
import tensorflow as tf
from PIL import Image as PImage

# from upload_image.forms.upload_image import ImageUploadForm
# from upload_image.utils.handle_upload_file import handle_uploaded_file

# class ImageUploadForm(forms.Form):
#     image = forms.ImageField()


# Create your views here.

class CreateImageView(CreateView):
    model = Image
    form_class = ImageUploadForm
    template_name = 'index.html'
    success_url = reverse_lazy('upload.html')


# def index(request):
#     return render(request, 'index.html')


def upload(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            imagefile = request.FILES['image']
            url_location = os.path.join(settings.MEDIA_ROOT, 'images')
            fs = FileSystemStorage(location=f'{url_location}')
            del_image_list = list(os.listdir(settings.IMAGE_ROOT))
            if len(del_image_list) > 0:
                del_img = del_image_list[0]
                fs.delete(del_img)
            filename = fs.save(imagefile.name, imagefile)
            uploaded_file_url = fs.url(filename)
            return render(request, 'upload.html', {'uploaded_file_url': uploaded_file_url})

    return HttpResponseForbidden('Allowed only by POST')


def result(request):
    model = settings.MODEL
    
    predictor = os.path.join(settings.PREDICTOR_ROOT, 'shape_predictor_68_face_landmarks.dat')
    # print('1')
    test_image_list = list(os.listdir(settings.IMAGE_ROOT))
    # print('2')
    print(test_image_list)
    test_img = test_image_list[-1]
    # test_img = os.path.join(settings.MEDIA_ROOT, test_img)
    # print('3')
    img = os.path.join(settings.IMAGE_ROOT, f'{test_img}')
    # print(img)
    # print('4')

    face_detector = dlib.get_frontal_face_detector()
    # print('5')
    image = io.imread(img)
    # print('6')
    detected_face = face_detector(image, 3)
    # print('7')
    
    predictor_model = predictor
    # print('8')
    face_aligner = openface.AlignDlib(predictor_model)
    # print('9')

    for i, rect in enumerate(detected_face):
        crop_area = (rect.left(), rect.top(), rect.right(), rect.bottom())
    # print('10')

    alignedFace = face_aligner.align(224, image, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    # print('11')
    im = numpy.reshape(alignedFace, (1,224,224,3))/255
    # print('12')
    # print(im)
    face_name = model.predict(im)
    # print('13')
    train_gen = ['AhnJaeHyun', 'AhnYongJun', 'AnSungKi', 'Andy', 'BaeJongwook', 'BoGyum', 'ByunYoHan', 'ChaEunWoo', 'ChaSeungWon', 'ChaTaeHyun', 'ChaeYoungIn', 'ChoBoa', 'ChoSeungWoo', 'ChoiSooJong', 'ChoiYeoJin', 'ChoiYooJung', 'ChuSangMi', 'ChungHa', 'Din', 'GongYoo', 'HaChunHwa', 'HaSeokJin', 'HanBoBae', 'HanChaeAh', 'HanChaeYoung', 'HanHaeJin(model)', 'HanJiHae', 'HanYaeSeul', 'HanYeoWoon', 'HeoGaYoon', 'HongSuHyun', 'HyaeRi', 'IRIn', 'JangNaRa', 'JangWonYoung', 'JeongKook', 'JiChangWook', 'JiMin', 'JiSangRyul', 'JiSeokJin', 'JiSooWon', 'JungIlWoo', 'JungTaeWoo', 'KanMiYeon', 'KangHoDong', 'KimBum', 'KimDaHyeon', 'KimDongHyun', 'KimHaOn', 'KimHyangKi', 'KimRaeWon', 'KimSooHyun', 'KimSooMi', 'KimSungRyung', 'KimYooJung', 'KoASung', 'KoJooWon', 'KongHyoJin', 'KooHaeSun', 'Kunst', 'KyunMiri', 'LeeKyungKyu', 'LeeSeungKi', 'LeeSeungWoo', 'LeeSungKyung', 'MinHyoRin', 'MoMo', 'NaYeon', 'NamHeeSeok', 'NamJuHyeok', 'Nancy', 'OnYoo', 'ParkBoGum', 'ParkBoYoung', 'ParkCheolMin', 'ParkEunBin', 'ParkHyoSin', 'ParkJiSung', 'ParkJunKyu', 'ParkJungHoon', 'ParkMiSeon', 'ParkSinHae', 'ParkSinYang', 'SaNa', 'SeoInKook', 'SeoKangJun', 'SeongSo', 'SeulKi', 'SoJin', 'SonHeungMin', 'SongHaeKyo', 'SongJungKi', 'SoyHyun', 'SulHyun', 'SunMi', 'SungJae', 'TaeJinA', 'TaeYeon', 'TakJaeHoon', 'V', 'YangJiWon', 'YeJiWon', 'YeoJinKoo', 'YeonWoo', 'YoanaJung', 'YooBin', 'YooHaeJin', 'YooInYeong', 'YooJunSang', 'YooSeungHo', 'YoonEunHye', 'YoonSiYun', 'YounAh', 'YounHa', 'YuRa', 'iu']
    # print('14')
    # rank1 = os.path.join(settings.OUTPUT_ROOT, f'{list(train_gen)[numpy.argmax(face_name)]}.jpg')
    rank1 = list(train_gen)[numpy.argmax(face_name)]
    # print('15')
    rank5 = sorted(face_name[0], reverse=True)[:5]
    # print('16')
    
    return render(request, 'result.html', {'rank1':rank1, 'rank5':rank5})