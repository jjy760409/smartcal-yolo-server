import base64
import io

import numpy as np
from PIL import Image

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import YOLO

# -----------------------------
# 1. FastAPI 기본 설정
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 나중에 smartcal-ai.com 으로 제한해도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# 2. 요청 바디 모델 (프론트 → 서버)
# -----------------------------
class ImageData(BaseModel):
    image: str   # base64 문자열


# -----------------------------
# 3. YOLO 모델 로딩
# -----------------------------
MODEL_PATH = "yolov8n.pt"  # 나중에 yolov8m.pt 로 바꿔도 됨
model = YOLO(MODEL_PATH)
names = model.names  # 클래스 이름 딕셔너리 (id → name)


# -----------------------------
# 4. 확장된 칼로리/정보 테이블
#    - YOLO COCO 클래스 이름 기준
#    - 각 항목에 한식/양식/일식/중식/디저트 등 정보 추가
# -----------------------------
# 키: YOLO 클래스 이름
# -----------------------------
# 4. 확장된 칼로리 테이블
#    - YOLO COCO 클래스 이름 기준 (영문 key)
#    - name: 한국어 이름 (+ 1인분 설명)
#    - kcal: 대략적인 칼로리
#    - cuisine: 한식/서양/일식/중식/동남아/중동/디저트/과일/해산물 등
#    - category: 밥/면/찌개/튀김/빵/디저트/과일 등
#    - portion: 기준량(1인분, 1조각 등)
# -----------------------------
CALORIE_TABLE = {
    # 🍙 밥/면/주식 25
    "rice": ("햅쌀밥 1공기", 310),
    "gimbap": ("김밥 1줄", 320),
    "ramen": ("라면 1봉지", 500),
    "sushi": ("초밥(모듬 8pcs)", 420),
    "noodle": ("국수 1그릇", 390),
    "fried rice": ("볶음밥 1접시", 600),
    "udon": ("우동 1그릇", 470),
    "rice cake": ("떡 1개", 50),
    "cold noodles": ("냉면 1그릇", 480),
    "pork belly rice": ("제육덮밥 1그릇", 770),
    "bibimbap": ("비빔밥 1그릇", 670),
    "kimchi fried rice": ("김치볶음밥", 720),
    "chicken mayo": ("치킨마요", 820),
    "tteokbokki": ("떡볶이 1인분", 550),
    "kalguksu": ("칼국수 1그릇", 590),
    "jajang": ("자장면 1그릇", 730),
    "jjamppong": ("짬뽕 1그릇", 780),
    "mandu": ("만두 5개", 350),
    "ramyun": ("라면 1개", 500),
    "jjigae rice": ("찌개 + 밥 세트", 850),
    "pork cutlet": ("돈까스 1인분", 850),
    "soba": ("소바 1그릇", 440),
    "onigiri": ("삼각김밥 1개", 190),
    "omelet rice": ("오므라이스 1접시", 780),
    "toast": ("토스트 1개", 420),

    # 🍖 고기/구이/튀김류 22
    "fried chicken": ("치킨 1조각", 250),
    "pork belly": ("삼겹살 100g", 320),
    "bulgogi": ("불고기 1접시", 580),
    "dakgalbi": ("닭갈비 1인분", 650),
    "jokbal": ("족발 200g", 540),
    "bossam": ("보쌈 1인분", 600),
    "galbi": ("갈비구이 200g", 740),
    "yangnyeom chicken": ("양념치킨 1조각", 300),
    "fried shrimp": ("새우튀김 1개", 90),
    "haemul pajeon": ("해물파전 1조각", 200),
    "sundae": ("순대 1인분", 550),
    "pajeon": ("파전 1조각", 220),
    "fried dumpling": ("군만두 5개", 400),
    "tteokgalbi": ("떡갈비 1개", 280),
    "gamjatang meat": ("감자탕 고기 1인분", 620),
    "kkochi": ("닭꼬치 1개", 160),
    "yangnyeom pork": ("제육볶음 1접시", 620),
    "fried pork": ("탕수육 10조각", 720),
    "chicken skewer": ("닭꼬치", 150),
    "deep fried pork": ("돈가스 1개", 850),
    "jeyuk": ("제육 1접시", 700),
    "godeungeo": ("고등어 구이", 330),

    # 🍲 찌개/탕/국물 18
    "kimchi stew": ("김치찌개 1인분", 520),
    "doenjang stew": ("된장찌개 1인분", 480),
    "soft tofu stew": ("순두부찌개 1인분", 620),
    "army stew": ("부대찌개 1인분", 780),
    "yukgaejang": ("육개장 1그릇", 560),
    "gukbap": ("국밥 1그릇", 750),
    "fish cake soup": ("오뎅탕", 210),
    "sundaeguk": ("순대국밥", 890),
    "seolleongtang": ("설렁탕 1그릇", 410),
    "galbitang": ("갈비탕 1그릇", 580),
    "haejangguk": ("해장국", 520),
    "dakgaejang": ("닭개장", 480),
    "maeuntang": ("매운탕", 350),
    "jjukkumi stew": ("쭈꾸미찌개", 450),
    "tteokguk": ("떡국", 500),
    "fish stew": ("생선찌개", 410),
    "janchi guksu": ("잔치국수", 400),
    "tomato stew": ("토마토스튜(한국형)", 460),

    # 🍱 반찬/김치 15
    "kimchi": ("배추김치 1접시", 60),
    "jangjorim": ("장조림", 210),
    "tteok": ("가래떡 1조각", 70),
    "anchovy": ("멸치볶음", 160),
    "spinach": ("시금치무침", 50),
    "bean sprout": ("콩나물무침", 45),
    "namul": ("나물모둠", 140),
    "egg roll": ("계란말이 1조각", 80),
    "jjajangbap side": ("단무지", 20),
    "kim side": ("김(3장)", 15),
    "potato salad": ("감자샐러드", 200),
    "pickled radish": ("치킨무", 10),
    "soup side": ("국물 반찬", 40),
    "myeolchi": ("멸치", 80),
    "sausage veg": ("소시지야채볶음", 270),
}



# -----------------------------
# 5. base64 → PIL.Image 변환 함수
# -----------------------------
def decode_base64_image(b64_str: str) -> Image.Image:
    # "data:image/jpeg;base64,..." 형식일 수도 있고
    # 순수 base64 문자열일 수도 있어서 , 기준으로 한 번 잘라줌
    if "," in b64_str:
        _, b64_str = b64_str.split(",", 1)

    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


# -----------------------------
# 6. /predict 엔드포인트 (프론트에서 호출)
# -----------------------------
@app.post("/predict")
def predict(data: ImageData):
    """
    1) base64 이미지를 디코딩하고
    2) YOLO로 음식 후보를 찾고
    3) CALORIE_TABLE 과 매칭해서
       items + totalCalories 형태로 돌려줌
    """
    # 1. 이미지 디코딩
    try:
        img = decode_base64_image(data.image)
    except Exception as e:
        return {"success": False, "error": f"이미지 디코딩 실패: {e}"}

    # 2. YOLO 추론
    try:
        np_img = np.array(img)
        results = model(np_img)[0]  # 첫 번째 결과만 사용
    except Exception as e:
        return {"success": False, "error": f"YOLO 추론 중 오류: {e}"}

    items = []

    # 3. 감지된 박스들 순회
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 신뢰도 너무 낮으면 패스
            if conf < 0.35:
                continue

            cls_name = names.get(cls_id, "")

            # 우리가 칼로리 테이블에 등록한 클래스만 사용
            if cls_name in CALORIE_TABLE:
                info = CALORIE_TABLE[cls_name]
                items.append(
                    {
                        "foodName": info["name"],
                        "calories": info["kcal"],
                        "cuisine": info["cuisine"],    # 한식/양식/일식/중식/기타
                        "category": info["category"],  # 주식/반찬/디저트/과일 등
                        "portion": info["portion"],    # 기본 1인분 설명
                        "conf": round(conf, 3),
                    }
                )

    # 4. 아무 음식도 못 찾았을 때
    if not items:
        return {
            "items": [],
            "totalCalories": 0,
            "note": "YOLO가 명확한 음식 객체를 찾지 못했습니다. 음식이 화면 중앙에 잘 보이도록 다시 촬영해 주세요.",
        }

    # 5. 총 칼로리 계산
    total_kcal = sum(item["calories"] for item in items)

    # 6. 안내 메시지 만들기 (추가 정보 포함)
    detail_lines = []
    for item in items:
        line = (
            f"• {item['foodName']} ≈ {item['calories']} kcal "
            f"(신뢰도 {item['conf']}, 분류: {item['cuisine']} / {item['category']}, 기준량: {item['portion']})"
        )
        detail_lines.append(line)

    note = (
        "YOLOv8 기반 자동 인식 결과입니다. 실제 음식 종류, 양, 조리법에 따라 칼로리는 달라질 수 있어요.\n"
        + "\n".join(detail_lines)
    )

    # 7. 프론트가 이해할 수 있는 형태로 반환
    return {
        "items": [
            {
                "foodName": item["foodName"],
                "calories": item["calories"],
                "cuisine": item["cuisine"],
                "category": item["category"],
                "portion": item["portion"],
            }
            for item in items
        ],
        "totalCalories": total_kcal,
        "note": note,
    }
