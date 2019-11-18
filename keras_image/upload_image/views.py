import os
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponseForbidden
from django import forms
from django.core.files.storage import FileSystemStorage
from django.conf import settings
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

def index(request):
    return render(request, 'upload_image/index.html')    


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
            return render(request, 'upload_image/upload.html', {'uploaded_file_url': uploaded_file_url})

    return HttpResponseForbidden('잘못된 접근입니다.')


def result(request):
    model = settings.MODEL
    
    predictor = os.path.join(settings.PREDICTOR_ROOT, 'shape_predictor_68_face_landmarks.dat')
    # print('1')
    test_image_list = list(os.listdir(settings.IMAGE_ROOT))
    # print('2')
    print(11111, test_image_list)
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
    face_name = list(model.predict(im)[0])
    # print('13')
    names = ['AhnJaeHyun', 'AhnYongJun', 'AnSungKi', 'Andy', 'BaeJongwook', 'BoGyum', 'ByunYoHan', 'ChaEunWoo', 'ChaSeungWon', 'ChaTaeHyun', 'ChaeYoungIn', 'ChoBoa', 'ChoSeungWoo', 'ChoiSooJong', 'ChoiYeoJin', 'ChoiYooJung', 'ChuSangMi', 'ChungHa', 'Din', 'GongYoo', 'HaChunHwa', 'HaSeokJin', 'HanBoBae', 'HanChaeAh', 'HanChaeYoung', 'HanHaeJin(model)', 'HanJiHae', 'HanYaeSeul', 'HanYeoWoon', 'HeoGaYoon', 'HongSuHyun', 'HyaeRi', 'IRIn', 'JangNaRa', 'JangWonYoung', 'JeongKook', 'JiChangWook', 'JiMin', 'JiSangRyul', 'JiSeokJin', 'JiSooWon', 'JungIlWoo', 'JungTaeWoo', 'KanMiYeon', 'KangHoDong', 'KimBum', 'KimDaHyeon', 'KimDongHyun', 'KimHaOn', 'KimHyangKi', 'KimRaeWon', 'KimSooHyun', 'KimSooMi', 'KimSungRyung', 'KimYooJung', 'KoASung', 'KoJooWon', 'KongHyoJin', 'KooHaeSun', 'Kunst', 'KyunMiri', 'LeeKyungKyu', 'LeeSeungKi', 'LeeSeungWoo', 'LeeSungKyung', 'MinHyoRin', 'MoMo', 'NaYeon', 'NamHeeSeok', 'NamJuHyeok', 'Nancy', 'OnYoo', 'ParkBoGum', 'ParkBoYoung', 'ParkCheolMin', 'ParkEunBin', 'ParkHyoSin', 'ParkJiSung', 'ParkJunKyu', 'ParkJungHoon', 'ParkMiSeon', 'ParkSinHae', 'ParkSinYang', 'SaNa', 'SeoInKook', 'SeoKangJun', 'SeongSo', 'SeulKi', 'SoJin', 'SonHeungMin', 'SongHaeKyo', 'SongJungKi', 'SoyHyun', 'SulHyun', 'SunMi', 'SungJae', 'TaeJinA', 'TaeYeon', 'TakJaeHoon', 'V', 'YangJiWon', 'YeJiWon', 'YeoJinKoo', 'YeonWoo', 'YoanaJung', 'YooBin', 'YooHaeJin', 'YooInYeong', 'YooJunSang', 'YooSeungHo', 'YoonEunHye', 'YoonSiYun', 'YounAh', 'YounHa', 'YuRa', 'iu']
    name = ['안재현', '안용준', '안성기', '앤디', '배종욱', '보겸', '변요한', '차은우', '차승원', '차태현', '채영인', '조보아', '조승우', '최수종', '최여진', '최유정', '추상미', '청하', '딘', '공유', '하춘화', '하석진', '한보배', '한채아', '한채영', '한혜진', '한지혜', '한예슬', '한여운', '허가윤', '홍수현', '혜리', '아이린', '장나라', '장원영', '정국', '지창욱', '지민', '지상렬', '지석진', '지수원', '정일우', '정태우', '간미연', '강호동', '김범', '김다현', '김동현', '김하온', '김향기', '김래원', '김수현', '김수미', '김성령', '김유정', '고아성', '고주원', '공효진', '구혜선', '코드쿤스트', '견미리', '이경규', '이승기', '이승우', '이성경', '민효린', '모모', '나연', '남희석', '남주혁', '낸시', '온유', '박보검', '박보영', '박철민', '박은빈', '박효신', '박지성', '박준규', '박중훈', '박미선', '박신혜', '박신양', '사나', '서인국', '서강준', '성소', '슬기', '소진', '손흥민', '송혜교', '송중기', '소이현', '설현', '선미', '육성재', '태진아', '태연', '탁재훈', '뷔', '양지원', '예지원', '여진구', '연우', '정려원', '유빈', '유혜진', '유인영', '유준상', '유승호', '윤은혜', '윤시윤', '윤아', '윤하', '유라', '아이유']
    # print('14')
    # rank1 = os.path.join(settings.OUTPUT_ROOT, f'{list(names)[numpy.argmax(face_name)]}.jpg')
    rank1 = names[face_name.index(sorted(face_name, reverse=True)[0])]
    rank2 = names[face_name.index(sorted(face_name, reverse=True)[1])]
    rank3 = names[face_name.index(sorted(face_name, reverse=True)[2])]
    rank4 = names[face_name.index(sorted(face_name, reverse=True)[3])]
    rank5 = names[face_name.index(sorted(face_name, reverse=True)[4])]

    name1 = name[face_name.index(sorted(face_name, reverse=True)[0])]
    name2 = name[face_name.index(sorted(face_name, reverse=True)[1])]
    name3 = name[face_name.index(sorted(face_name, reverse=True)[2])]
    name4 = name[face_name.index(sorted(face_name, reverse=True)[3])]
    name5 = name[face_name.index(sorted(face_name, reverse=True)[4])]

    probs1 = sorted(face_name, reverse=True)[0] * 100
    probs2 = sorted(face_name, reverse=True)[1] * 100
    probs3 = sorted(face_name, reverse=True)[2] * 100
    probs4 = sorted(face_name, reverse=True)[3] * 100
    probs5 = sorted(face_name, reverse=True)[4] * 100

    res = {
        'MEDIA_URL':settings.MEDIA_URL,
        'rank1':rank1,
        'rank2':rank2,
        'rank3':rank3,
        'rank4':rank4,
        'probs1':probs1,
        'probs2':probs2,
        'probs3':probs3,
        'probs4':probs4,
        'input_image':test_img,
        'name1': name1,
        'name2': name2,
        'name3': name3,
        'name4': name4,
    }
    print(settings.STATIC_URL)
    # print('16')
    # print(rank5)
    # print(probs1)
    return render(request, 'upload_image/result.html', res)