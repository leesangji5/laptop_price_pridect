import pandas as pd
import joblib
import easyocr
import cv2
import re

# image path
image_path = 'd.jpg'

# processor_name i3, i5, i7, i9에서 숫자
# processor_gnrtn 세대 
# ram_gb 렘 용량
# ram_type DDR3, DDR4, LPDDR4등 숫자
# storage 저장공간 용량
# os 윈도우 여부로 판단함 ex) 리눅스, macos여도 0으로 하고, 윈도우만, 1로 판단
# 리눅스, macos는 기본적으로는 무료이기 때문에 유료인 윈도우만 1로 판단
# graphic_card_gb 그래픽카드 vram

def read_text_from_image(image_path):
    # EasyOCR 객체 생성
    reader = easyocr.Reader(['en', 'ko'])

    # 이미지 불러오기
    image = cv2.imread(image_path)
    # 이미지 RGB 형식으로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 글자 인식
    result = reader.readtext(image_rgb)

    # 필요한 정보 추출을 위한 변수
    graphic_card = False
    storage_check = 0
    price_check = False

    # dictionary 생성
    matches = dict()
    matches["processor_name"] = ""
    matches["processor_gnrtn"] = ""
    matches["ram_gb"] = ""
    matches["ram_type"] = ""
    matches["storage"] = ""
    matches["os"] = ""
    matches["graphic_card_gb"] = ""
    matches["price"] = ""

    for detection in result:
        text = detection[1]
        confidence = detection[2]

        # 0을 O로 인식하는 경우 수정
        text = re.sub(r'(\d+)OGB', r'\g<1>0GB', text)
        # 5를 S로 인식하는 경우 수정
        text = re.sub(r"(LP)?DDRS", r"\1LPDDR5", text)

        # 인식된 글자에서 필요한 정보 추출
        if (re.search(r"i(\d+)", text)):
            matches["processor_name"] = re.findall(r"i(\d+)", text)[0]

        if (re.search(r"\d+(?=세대)", text)):
            matches["processor_gnrtn"] = re.findall(r"\d+(?=세대)", text)[0]

        if (re.search(r"램 용량|램용량|뱀 용량", text)):
            matches["ram_gb"] = re.findall(r"(\d+)GB", text)[0]

        if (re.search(r"DDR(\d+)", text)):
            matches["ram_type"] = re.findall(r"DDR(\d+)", text)[0]
        elif (re.search(r"LPDDR(\d+)", text)):
            matches["ram_type"] = re.findall(r"LPDDR(\d+)", text)[0]
        elif (re.search(r"DDR", text)):
            matches["ram_type"] = re.findall(r"DDR(\d+)", text)[0]

        if (re.search(r"SSD", text) and storage_check == 0):
            storage_check += 1
        if (re.search(r"(\d+)GB", text) and storage_check == 1):
            matches["storage"] = re.findall(r"(\d+)GB", text)[0]
            storage_check += 1
        elif (re.search(r"(\d+)TB", text) and storage_check == 1):
            matches["storage"] = '{}'.format(int(re.findall(r"(\d+)TB", text)[0]) * 1000)
            storage_check += 1

        if (re.search(r"(윈도우|원도우|미포함|macOS|macoS|macOs|리눅스|리뉴스)", text)):
            matches["os"] = re.findall(r"(윈도우|원도우|미포함|macOS|macoS|macOs|리눅스|리뉴스)", text)[0]
            if matches["os"] == "원도우":
                matches["os"] = "윈도우"
            elif matches["os"] == "리뉴스":
                matches["os"] = "리눅스"
            elif matches["os"] == "macoS" or matches["os"] == "macOs":
                matches["os"] = "macOS"

        if (re.search(r"(외장|내장)그래픽", text)):
            if re.findall(r"(외장|내장)그래픽", text)[0] == "외장":
                graphic_card = True
        if (re.search(r"VRAM:(\d+GB)", text) and graphic_card):
            matches["graphic_card_gb"] = re.findall(r"VRAM:(\d+GB)", text)[0]

        if (re.search(r'\d{1,3}(?:,\d{3})*원', text) and price_check == False):
            price_match = re.search(r'\d{1,3}(?:,\d{3})*원', text)
            matches["price"] = re.sub(r'[,\s원]+', '', price_match[0])
            price_check = True

    # 그래픽 카드가 없는 경우
    if (graphic_card == False):
        matches["graphic_card_gb"] = "0"
    elif (re.search(r"GB", matches["graphic_card_gb"])):
        matches["graphic_card_gb"] = re.sub(r'[GB]+', '', matches["graphic_card_gb"])

    return matches

# 개발할 당시의 환율 2023/7/22
def pridict(image_data, error_range=100000, won2cent=0.078, cent2won=1283.44):

    # 가격 (단위: 센트)
    real_price = int(image_data['price'])*won2cent

    # 단위 변환
    input_money_lang = 'cent'
    output_money_lang = 'won'
    if output_money_lang == 'won':
        money_weight = cent2won / 100
    elif output_money_lang == 'dollar':
        money_weight = 1 / 100
    elif output_money_lang == 'cent':
        money_weight = 1

    # price는 target이므로 data에서 제외
    # os는 윈도우의 여부로 고려함
    data = dict()
    data['processor_name'] = [image_data['processor_name']]
    data['processor_gnrtn'] = [image_data['processor_gnrtn']]
    data['ram_gb'] = [image_data['ram_gb']]
    data['ram_type'] = [image_data['ram_type']]
    data['storage'] = [image_data['storage']]
    if re.search('윈도우', image_data['os']):
        data['os'] = [1]
    else:
        data['os'] = [0]
    data['graphic_card_gb'] = [image_data['graphic_card_gb']]

    # 데이터 출력
    print(data)

    # 학습된 모델 불러오기
    regressor = joblib.load('laptop_regression_model.pkl')

    # 예측
    pridicted_price = int(regressor.predict(pd.DataFrame(data))[0])

    # 단위 변환
    real_price = real_price * money_weight
    pridicted_price = pridicted_price * money_weight

    # 출력
    print('real price:', int(real_price), output_money_lang)
    print('pridicted price:', int(pridicted_price), output_money_lang)
    print('error range:', int(error_range), output_money_lang)
    print('error:',int(abs(real_price - pridicted_price)), output_money_lang)
    if real_price - pridicted_price + error_range > 0:
        print('buy')
    else:
        print('not buy')

image_data = read_text_from_image(image_path)
pridict(image_data)
