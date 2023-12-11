from ultralytics import YOLO

model = YOLO('best (1).pt')
results = model.predict(source='Test_images/20231211_164424.jpg', save=False, imgsz=(640, 640), conf=0.7)
boxes = []
path = str()
for result in results:
    boxes = list(result.boxes.cls)  # Boxes object for bbox cls outputs
    path = ''.join(str(result.path))


coins = {0: 10, 1: 20, 2: 50, 3: 100, 4: 200, 5: 500}

amount_coins = []

for i in boxes:
    amount_coins.append(coins[int(i)])

print(f'Image Path is {path}')
print(f'{sum(amount_coins)} AMD')

