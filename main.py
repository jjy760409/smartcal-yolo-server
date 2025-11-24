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
    allow_origins=["*"],   # 나중에 smartcal-ai.com 으로 변경 가능
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
MODEL_PATH = "yolov8n.pt"  # 나중에 yolov8m.pt 등으로 변경 가능
model = YOLO(MODEL_PATH)
names = model.names  # 클래스 이름 딕셔너리 (id → name)


# -----------------------------
# 4. 확장된 칼로리/정보 테이블
#    - key: YOLO 클래스 이름 또는 커스텀 클래스 이름
#    - foodName: 한국어 표시 이름
#    - calories: 대략적인 1인분 칼로리
#    - cuisine: 한식/일식/중식/디저트/음료 등
#    - category: 밥/면/국물/튀김/디저트/음료 등
#    - portion: 기준량 설명
#    - tags: 추가 태그(선택)
# -----------------------------
CALORIE_TABLE = {
    # =============================
    # 🍚 한식 - 밥/비빔밥
    # =============================
    "k_rice_basic": {
        "foodName": "쌀밥(1공기)",
        "calories": 300,
        "cuisine": "Korean",
        "category": "밥",
        "portion": "1공기(210g)",
        "tags": ["밥", "기본"],
    },
    "k_rice_brown": {
        "foodName": "현미밥(1공기)",
        "calories": 330,
        "cuisine": "Korean",
        "category": "밥",
        "portion": "1공기(210g)",
        "tags": ["건강"],
    },
    "k_bibimbap": {
        "foodName": "비빔밥(1그릇)",
        "calories": 550,
        "cuisine": "Korean",
        "category": "밥",
        "portion": "1그릇",
        "tags": ["정식"],
    },
    "k_kimchi_fried_rice": {
        "foodName": "김치볶음밥",
        "calories": 680,
        "cuisine": "Korean",
        "category": "볶음밥",
        "portion": "1그릇",
        "tags": ["볶음밥"],
    },
    "k_japgokbab": {
        "foodName": "잡곡밥",
        "calories": 350,
        "cuisine": "Korean",
        "category": "밥",
        "portion": "1공기",
        "tags": ["건강"],
    },
    "k_gimbap_basic": {
        "foodName": "김밥(1줄)",
        "calories": 320,
        "cuisine": "Korean",
        "category": "분식",
        "portion": "1줄",
        "tags": ["분식"],
    },
    "k_omurice": {
        "foodName": "오므라이스",
        "calories": 700,
        "cuisine": "Korean",
        "category": "밥",
        "portion": "1접시",
        "tags": ["어린이", "경양식"],
    },

    # =============================
    # 🍜 한식 - 면/분식
    # =============================
    "k_ramen": {
        "foodName": "라면(1봉지)",
        "calories": 500,
        "cuisine": "Korean",
        "category": "면",
        "portion": "1봉지 기준",
        "tags": ["간편"],
    },
    "k_tteokbokki_basic": {
        "foodName": "기본 떡볶이(1인분)",
        "calories": 550,
        "cuisine": "Korean",
        "category": "분식",
        "portion": "1인분",
        "tags": ["분식", "매운"],
    },
    "k_tteokbokki_cheese": {
        "foodName": "치즈 떡볶이",
        "calories": 680,
        "cuisine": "Korean",
        "category": "분식",
        "portion": "1인분",
        "tags": ["치즈", "분식"],
    },
    "k_bibim_naeng": {
        "foodName": "비빔냉면",
        "calories": 540,
        "cuisine": "Korean",
        "category": "면",
        "portion": "1그릇",
        "tags": ["여름"],
    },
    "k_plain_naeng": {
        "foodName": "물냉면",
        "calories": 460,
        "cuisine": "Korean",
        "category": "면",
        "portion": "1그릇",
        "tags": ["여름"],
    },

    # 튀김/분식 사이드
    "k_fried_squid": {
        "foodName": "오징어튀김(2개)",
        "calories": 320,
        "cuisine": "Korean",
        "category": "튀김",
        "portion": "2개",
        "tags": ["분식"],
    },
    "k_fried_shrimp": {
        "foodName": "새우튀김(2개)",
        "calories": 380,
        "cuisine": "Korean",
        "category": "튀김",
        "portion": "2개",
        "tags": ["분식"],
    },
    "k_bungeoppang": {
        "foodName": "붕어빵(2개)",
        "calories": 340,
        "cuisine": "Korean",
        "category": "간식",
        "portion": "2개",
        "tags": ["겨울간식"],
    },

    # =============================
    # 🍖 한식 - 고기/BBQ
    # =============================
    "k_samgyeopsal": {
        "foodName": "삼겹살(200g)",
        "calories": 780,
        "cuisine": "Korean",
        "category": "고기",
        "portion": "200g",
        "tags": ["구이"],
    },
    "k_galbi": {
        "foodName": "양념갈비(200g)",
        "calories": 890,
        "cuisine": "Korean",
        "category": "고기",
        "portion": "200g",
        "tags": ["단짠"],
    },
    "k_bulgogi": {
        "foodName": "불고기",
        "calories": 510,
        "cuisine": "Korean",
        "category": "고기",
        "portion": "1인분",
        "tags": ["정식"],
    },
    "k_jeyuk": {
        "foodName": "제육볶음",
        "calories": 650,
        "cuisine": "Korean",
        "category": "고기",
        "portion": "1인분",
        "tags": ["매운"],
    },

    # =============================
    # 🍲 한식 - 국/찌개
    # =============================
    "k_kimchi_stew": {
        "foodName": "김치찌개",
        "calories": 450,
        "cuisine": "Korean",
        "category": "찌개",
        "portion": "1인분",
        "tags": ["찌개"],
    },
    "k_soybean_paste": {
        "foodName": "된장찌개",
        "calories": 350,
        "cuisine": "Korean",
        "category": "찌개",
        "portion": "1인분",
        "tags": ["찌개"],
    },
    "k_sundae_soup": {
        "foodName": "순대국밥",
        "calories": 630,
        "cuisine": "Korean",
        "category": "국밥",
        "portion": "1그릇",
        "tags": ["국밥"],
    },
    "k_gamjatang": {
        "foodName": "감자탕",
        "calories": 700,
        "cuisine": "Korean",
        "category": "탕",
        "portion": "1인분",
        "tags": ["해장"],
    },
    "k_miyeok": {
        "foodName": "미역국",
        "calories": 210,
        "cuisine": "Korean",
        "category": "국",
        "portion": "1그릇",
        "tags": ["기본"],
    },

    # =============================
    # 🍣 일식 Japanese Food
    # =============================
    "sushi": {
        "foodName": "스시(접시 1개)",
        "calories": 150,
        "cuisine": "Japanese",
        "category": "밥",
        "portion": "초밥 2~3개 기준",
        "tags": ["일식"],
    },
    "ramen_jp": {
        "foodName": "일본 라멘(1그릇)",
        "calories": 550,
        "cuisine": "Japanese",
        "category": "면",
        "portion": "1그릇",
        "tags": ["국물"],
    },
    "udon_jp": {
        "foodName": "우동(1그릇)",
        "calories": 550,
        "cuisine": "Japanese",
        "category": "면",
        "portion": "1그릇",
        "tags": ["국물"],
    },
    "katsudon": {
        "foodName": "가츠동(1그릇)",
        "calories": 900,
        "cuisine": "Japanese",
        "category": "덮밥",
        "portion": "1그릇",
        "tags": ["덮밥"],
    },
    "takoyaki": {
        "foodName": "타코야끼(6개)",
        "calories": 350,
        "cuisine": "Japanese",
        "category": "간식",
        "portion": "6개",
        "tags": ["간식"],
    },

    # =============================
    # 🥡 중식 Chinese Food
    # =============================
    "jajangmyeon": {
        "foodName": "짜장면(1그릇)",
        "calories": 800,
        "cuisine": "Chinese",
        "category": "면",
        "portion": "1그릇",
        "tags": ["중식"],
    },
    "jjamppong": {
        "foodName": "짬뽕(1그릇)",
        "calories": 750,
        "cuisine": "Chinese",
        "category": "면",
        "portion": "1그릇",
        "tags": ["중식", "매운"],
    },
    "tangsuyuk": {
        "foodName": "탕수육(1인분)",
        "calories": 900,
        "cuisine": "Chinese",
        "category": "튀김",
        "portion": "1인분",
        "tags": ["중식"],
    },
    "fried_rice_cn": {
        "foodName": "중식 볶음밥(1그릇)",
        "calories": 720,
        "cuisine": "Chinese",
        "category": "볶음밥",
        "portion": "1그릇",
        "tags": ["중식"],
    },

    # =============================
    # 🍰 디저트 / 베이커리 Dessert
    # =============================
    "cake": {
        "foodName": "케이크(1조각)",
        "calories": 350,
        "cuisine": "Dessert",
        "category": "디저트",
        "portion": "1조각",
        "tags": ["디저트"],
    },
    "icecream": {
        "foodName": "아이스크림(1회 제공)",
        "calories": 250,
        "cuisine": "Dessert",
        "category": "디저트",
        "portion": "1스쿱 기준",
        "tags": ["간식"],
    },
    "donut": {
        "foodName": "도넛(1개)",
        "calories": 280,
        "cuisine": "Dessert",
        "category": "디저트",
        "portion": "1개",
        "tags": ["간식"],
    },
    "cookie": {
        "foodName": "쿠키(1개)",
        "calories": 80,
        "cuisine": "Dessert",
        "category": "디저트",
        "portion": "1개",
        "tags": ["간식"],
    },

    # =============================
    # 🧃 음료 Drinks
    # =============================
    "cola": {
        "foodName": "콜라(캔 1개)",
        "calories": 140,
        "cuisine": "Drink",
        "category": "탄산음료",
        "portion": "355ml",
        "tags": ["음료"],
    },
    "cider": {
        "foodName": "사이다(캔 1개)",
        "calories": 140,
        "cuisine": "Drink",
        "category": "탄산음료",
        "portion": "355ml",
        "tags": ["음료"],
    },
    "americano": {
        "foodName": "아메리카노(1잔)",
        "calories": 5,
        "cuisine": "Drink",
        "category": "커피",
        "portion": "1잔",
        "tags": ["저칼로리"],
    },
    "latte": {
        "foodName": "카페라떼(1잔)",
        "calories": 180,
        "cuisine": "Drink",
        "category": "커피",
        "portion": "1잔",
        "tags": ["우유"],
    },
    "milk_tea": {
        "foodName": "밀크티(1잔)",
        "calories": 300,
        "cuisine": "Drink",
        "category": "티",
        "portion": "1잔",
        "tags": ["디저트"],
    },
    "orange_juice": {
        "foodName": "오렌지주스(1잔)",
        "calories": 110,
        "cuisine": "Drink",
        "category": "주스",
        "portion": "1잔",
        "tags": ["과일주스"],
    },
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
                        "foodName": info["foodName"],
                        "calories": info["calories"],
                        "cuisine": info["cuisine"],
                        "category": info["category"],
                        "portion": info["portion"],
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
                "conf": item["conf"],
            }
            for item in items
        ],
        "totalCalories": total_kcal,
        "note": note,
    }
