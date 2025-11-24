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
    # 01. 대표 한식 단품
    "kimbap":         {"foodName": "김밥(1줄)",           "calories": 320},
    "gimbap":         {"foodName": "김밥(1줄)",           "calories": 320},
    "ramen":          {"foodName": "라면(1봉지)",         "calories": 500},
    "tteokbokki":     {"foodName": "떡볶이(1인분)",       "calories": 600},
    "sundae":         {"foodName": "순대(1인분)",         "calories": 450},
    "fried_chicken":  {"foodName": "치킨(한 조각)",       "calories": 250},
    "whole_chicken":  {"foodName": "치킨(뼈 있는 1마리)", "calories": 1800},
    "yangnyeom_chicken": {"foodName": "양념치킨(1조각)", "calories": 280},
    "dakgangjeong":   {"foodName": "닭강정(1인분)",       "calories": 700},

    # 02. 밥 / 국 / 찌개 / 덮밥
    "bibimbap":       {"foodName": "비빔밥(1그릇)",       "calories": 650},
    "kimchi_fried_rice": {"foodName": "김치볶음밥(1그릇)", "calories": 700},
    "fried_rice":     {"foodName": "볶음밥(1그릇)",       "calories": 680},
    "white_rice":     {"foodName": "쌀밥(1공기)",         "calories": 300},
    "brown_rice":     {"foodName": "현미밥(1공기)",       "calories": 280},
    "japchae_rice":   {"foodName": "잡채밥(1그릇)",       "calories": 750},
    "pork_cutlet_rice": {"foodName": "돈까스덮밥(1그릇)", "calories": 900},
    "gyudon_korean":  {"foodName": "소고기덮밥(1그릇)",   "calories": 820},

    "doenjang_jjigae": {"foodName": "된장찌개(1인분)",     "calories": 200},
    "kimchi_jjigae":   {"foodName": "김치찌개(1인분)",     "calories": 250},
    "soft_tofu_stew":  {"foodName": "순두부찌개(1인분)",   "calories": 300},
    "seaweed_soup":    {"foodName": "미역국(1인분)",       "calories": 120},
    "seolleongtang":   {"foodName": "설렁탕(1그릇)",       "calories": 450},
    "galbitang":       {"foodName": "갈비탕(1그릇)",       "calories": 550},

    # 03. 면 / 분식
    "jajangmyeon":      {"foodName": "짜장면(1그릇)",      "calories": 800},
    "jjamppong":        {"foodName": "짬뽕(1그릇)",        "calories": 750},
    "cold_noodles":     {"foodName": "냉면(1그릇)",        "calories": 550},
    "bibim_naengmyeon": {"foodName": "비빔냉면(1그릇)",    "calories": 650},
    "kalguksu":         {"foodName": "칼국수(1그릇)",      "calories": 650},
    "udon":             {"foodName": "우동(1그릇)",        "calories": 550},
    "rabokki":          {"foodName": "라볶이(1인분)",      "calories": 700},
    "guksu":            {"foodName": "잔치국수(1그릇)",    "calories": 550},

    # 04. 고기 / 구이
    "samgyeopsal":     {"foodName": "삼겹살(100g)",        "calories": 520},
    "samgyeopsal_set": {"foodName": "삼겹살(1인분, 200g)", "calories": 1040},
    "bulgogi":         {"foodName": "소불고기(1인분)",     "calories": 550},
    "dakgalbi":        {"foodName": "닭갈비(1인분)",       "calories": 700},
    "galbi":           {"foodName": "돼지갈비(1인분)",     "calories": 800},
    "bossam":          {"foodName": "보쌈(1인분)",         "calories": 650},
    "jeyuk_bokkeum":   {"foodName": "제육볶음(1인분)",     "calories": 700},
    "soondae_guk":     {"foodName": "순댓국(1그릇)",       "calories": 650},

    # 05. 전 / 튀김
    "kimchi_jeon":     {"foodName": "김치전(1장)",         "calories": 300},
    "pajeon":          {"foodName": "파전(1장)",           "calories": 450},
    "haemul_pajeon":   {"foodName": "해물파전(1장)",       "calories": 550},
    "fried_shrimp":    {"foodName": "새우튀김(1개)",       "calories": 80},
    "fried_mandu":     {"foodName": "군만두(1개)",         "calories": 70},
    "steamed_mandu":   {"foodName": "찐만두(1개)",         "calories": 50},

    # 06. 반찬 / 사이드
    "kimchi":          {"foodName": "배추김치(소접시)",    "calories": 25},
    "kkakdugi":        {"foodName": "깍두기(소접시)",      "calories": 30},
    "egg_roll":        {"foodName": "계란말이(조각 1개)",  "calories": 60},
    "fried_egg":       {"foodName": "계란후라이(1개)",     "calories": 90},
    "cheese_slice":    {"foodName": "슬라이스 치즈(1장)",  "calories": 70},
    "sausage_pan":     {"foodName": "소시지볶음(소접시)",  "calories": 180},
    "fishcake":        {"foodName": "어묵볶음(소접시)",    "calories": 150},

    # 07. 한식 디저트 / 기타
    "hotteok":         {"foodName": "호떡(1개)",           "calories": 230},
    "bungeoppang":     {"foodName": "붕어빵(1개)",         "calories": 180},
    "injeolmi":        {"foodName": "인절미(조각 1개)",    "calories": 70},
    "yakgwa":          {"foodName": "약과(1개)",           "calories": 130},
    "sikhye":          {"foodName": "식혜(컵 1잔)",        "calories": 120},
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
