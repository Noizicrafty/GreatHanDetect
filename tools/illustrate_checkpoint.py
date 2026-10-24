from pathlib import Path
import json
import random
import hashlib
import os
os.environ['MPLBACKEND'] = 'Agg'
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageOps

import argparse
parser = argparse.ArgumentParser(description='Illustrate an existing YOLO checkpoint. Inference only; never starts training.')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--test-list', required=True)
parser.add_argument('--out', default='results/checkpoint_examples')
args = parser.parse_args()
REPO = Path(__file__).resolve().parent.parent
OUT = Path(args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'individual').mkdir(exist_ok=True)
WEIGHTS = Path(args.checkpoint).resolve()
ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
epoch = int(ckpt['epoch']) + 1
assert epoch >= 18, epoch
del ckpt
model = YOLO(str(WEIGHTS))
names = model.names
test_paths = sorted(p.strip() for p in Path(args.test_list).read_text(encoding='utf-8').splitlines() if p.strip())
samples = random.Random(42).sample(test_paths, 9)
colors = {0: '#F2AC32', 1: '#39B9E2', 2: '#FC666A', 3: '#BD8DF5'}
FONT = next((str(p) for p in [Path('C:/Windows/Fonts/segoeui.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.exists()), 'DejaVuSans.ttf')
BOLD = next((str(p) for p in [Path('C:/Windows/Fonts/segoeuib.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')] if p.exists()), FONT)
def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)

def draw_boxes(image, predictions, scale=1.0, offset=(0, 0), clip=None):
    draw = ImageDraw.Draw(image)
    for p in sorted(predictions, key=lambda x: x['class_id'] != 1):
        box = p['xyxy']
        if clip and (box[2] <= clip[0] or box[0] >= clip[2] or box[3] <= clip[1] or box[1] >= clip[3]):
            continue
        x1,y1,x2,y2 = [box[i] * scale + offset[i % 2] for i in range(4)]
        color = colors[p['class_id']]
        draw.rectangle((x1,y1,x2,y2), outline=color, width=3)
        label = f"{names[p['class_id']]} {p['confidence']:.2f}"
        f = font(19, True)
        bbox = draw.textbbox((0, 0), label, font=f)
        tw, th = bbox[2], bbox[3] - bbox[1] + 10
        tx = min(max(0, x1), max(0, image.width - tw - 12))
        ty = min(max(0, y1 - th), image.height - th)
        draw.rectangle((tx,ty,tx+tw+12,ty+th), fill=color)
        draw.text((tx+6,ty-1),label,font=f,fill='#0C1728')

records = []
cards = []
for i, path in enumerate(samples, 1):
    src = Path(path)
    image = Image.open(src).convert('RGB')
    result = model.predict(source=str(src), imgsz=640, conf=0.25, iou=0.7, device=0 if torch.cuda.is_available() else 'cpu', verbose=False)[0]
    predictions = [{'class_id': int(c), 'class_name': names[int(c)], 'confidence': float(s), 'xyxy': [float(t) for t in b]} for b,s,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
    gt_path = src.parent.parent / 'labels' / (src.stem + '.txt')
    gt = []
    for line in gt_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        c,x,y,w,h = map(float,line.split())
        gt.append({'class_id': int(c), 'xyxy': [(x-w/2)*image.width,(y-h/2)*image.height,(x+w/2)*image.width,(y+h/2)*image.height]})
    record = {'sample_id': i, 'source': str(src), 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'size': list(image.size), 'predictions': predictions, 'ground_truth': gt}
    records.append(record)
    # Letterbox the source, then draw the real predicted coordinates at the same scale.
    ratio = min(600 / image.width, 430 / image.height)
    scaled = image.resize((round(image.width*ratio),round(image.height*ratio)),Image.Resampling.LANCZOS)
    panel = Image.new('RGB',(620,505),'#152137')
    dx = (620-scaled.width)//2
    dy = 48 + (430-scaled.height)//2
    panel.paste(scaled,(dx,dy))
    draw_boxes(panel,predictions,ratio,(dx,dy))
    d = ImageDraw.Draw(panel)
    d.rectangle((0,0,620,44),fill='#152137')
    d.text((14,7),f"Тест #{i:02}  ·  детекций: {len(predictions)}",font=font(23,True),fill='white')
    if not predictions:
        d.text((14,475),'При conf ≥ 0,25 объекты не обнаружены',font=font(17),fill='#F2AC32')
    panel.save(OUT / 'individual' / f'{i:02}_prediction.png')
    full = image.copy()
    draw_boxes(full,predictions)
    full.save(OUT / 'individual' / f'{i:02}_full_resolution.jpg',quality=95)
    cards.append(panel)

sheet = Image.new('RGB',(1920,1740),'#0A1425')
d = ImageDraw.Draw(sheet)
d.text((32,22),f'Реальные предсказания модели · {epoch} эпох',font=font(45,True),fill='white')
d.text((34,82),'YOLOv8n · last.pt · тестовая выборка · 9 случайных изображений, seed=42',font=font(25),fill='#BDCADC')
d.text((34,122),'Сигарета — жёлтый   |   Человек — голубой   |   Дым — красный   |   Вейп — фиолетовый',font=font(23),fill='#BDCADC')
for i,card in enumerate(cards):
    sheet.paste(card,(20+(i%3)*630,175+(i//3)*515))
d.text((30,1720-20),'Рамки и числа — выход модели (conf ≥ 0,25). Это детекция объектов, не подтверждение эпизода курения.',font=font(21),fill='#BDCADC')
sheet.save(OUT / f'01_real_predictions_{epoch}_epochs.png')

# Six deterministic close-ups centered on annotated cigarette/smoke objects.
details = []
for rec in records:
    candidates = [g for g in rec['ground_truth'] if g['class_id'] in (0,2,3)]
    if not candidates:
        continue
    gt = candidates[0]
    image = Image.open(rec['source']).convert('RGB')
    x1,y1,x2,y2 = gt['xyxy']
    cx,cy = (x1+x2)/2,(y1+y2)/2
    side = min(max((x2-x1)*4,(y2-y1)*4,150),max(image.size))
    left,top = max(0,min(cx-side/2,image.width-side)),max(0,min(cy-side/2,image.height-side))
    cropbox = (int(left),int(top),min(image.width,int(left+side)),min(image.height,int(top+side)))
    crop = image.crop(cropbox)
    ratio = min(600/crop.width,430/crop.height)
    scaled = crop.resize((round(crop.width*ratio),round(crop.height*ratio)),Image.Resampling.LANCZOS)
    card = Image.new('RGB',(620,510),'#152137')
    dx,dy = (620-scaled.width)//2,52+(430-scaled.height)//2
    card.paste(scaled,(dx,dy))
    draw_boxes(card,rec['predictions'],ratio,(dx-cropbox[0]*ratio,dy-cropbox[1]*ratio),cropbox)
    d=ImageDraw.Draw(card)
    d.rectangle((0,0,620,48),fill='#152137')
    d.text((14,8),f"Тест #{rec['sample_id']:02} · увеличенный фрагмент",font=font(22,True),fill='white')
    d.rectangle((0,482,620,510),fill='#152137')
    d.text((14,483),'Область выбрана по GT; рамки — только предсказания',font=font(18),fill='#BDCADC')
    details.append(card)
    if len(details)==6:
        break
detail_sheet=Image.new('RGB',(1920,1230),'#0A1425')
d=ImageDraw.Draw(detail_sheet)
d.text((32,22),'Объекты курения крупным планом',font=font(45,True),fill='white')
d.text((34,84),f'Та же модель после {epoch} эпох · те же тестовые изображения · без изменения предсказаний',font=font(25),fill='#BDCADC')
for i,card in enumerate(details):
    detail_sheet.paste(card,(20+(i%3)*630,145+(i//3)*520))
d.text((30,1195),'Отсутствие рамки означает пропуск при пороге 0,25. Увеличение сделано после инференса.',font=font(22),fill='#BDCADC')
detail_sheet.save(OUT / f'02_object_closeups_{epoch}_epochs.png')
manifest={'checkpoint':str(WEIGHTS),'checkpoint_sha256':hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),'completed_epochs':epoch,'checkpoint_epoch_zero_based':epoch-1,'inference_only':True,'training_resumed':False,'backend':'PyTorch CUDA' if torch.cuda.is_available() else 'PyTorch CPU','imgsz':640,'confidence_threshold':0.25,'nms_iou':0.7,'sample_selection':'random.Random(42).sample(sorted(test_paths), 9)','dataset_test_count':len(test_paths),'classes':names,'records':records}
(OUT/'predictions_and_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'README_RU.md').write_text(f'''# Иллюстрации предсказаний обученной модели

Модель: YOLOv8n, checkpoint `last.pt`, завершено {epoch} эпох (внутреннее поле epoch={epoch-1}).
Это сохранённые веса прерванного тренировочного прогона, а не новая тренировка.

Проведён только инференс: PyTorch, размер входа 640, conf=0.25, NMS IoU=0.7.
Девять изображений выбраны из тестовой выборки фиксированным seed=42 до просмотра предсказаний.
Рамки и confidence взяты непосредственно из выхода модели. На втором листе область увеличения выбрана по разметке, но сама разметка не нарисована как предсказание.
Наличие сигареты или дыма ещё не означает подтверждённый временной эпизод курения.
Примеры не являются измерением точности всей выборки или проверкой работы TPU.

Источник изображений: Richie Lab / smoking-tasfx, Roboflow, версия 4; лицензия CC BY 4.0 согласно метаданным экспорта.
https://universe.roboflow.com/richie-lab/smoking-tasfx/dataset/4

Сырые предсказания, пути, хеши и параметры: `predictions_and_provenance.json`.
Отдельные кадры полного разрешения: `individual/`.
''',encoding='utf-8')
print(json.dumps({'output':str(OUT),'epochs':epoch,'images':len(records),'detections':sum(len(x['predictions']) for x in records)},ensure_ascii=False))
