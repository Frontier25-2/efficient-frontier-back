# frontier.py

from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from app.services.price_service import fetch_price_history
# [수정] compute_frontier_markers 함수 import 추가
from app.models.quant_model_modules import compute_efficient_frontier, compute_frontier_markers

bp = Blueprint("frontier", __name__)

@bp.route("/api/frontier", methods=["POST"])
def get_frontier():
    data = request.get_json()

    codes = data.get("codes")
    start = data.get("start")
    end = data.get("end")

    if not codes or len(codes) < 2:
        return jsonify({"error": "종목 최소 2개 필요"}), 400

    # 1. 데이터 수집 및 날짜 인덱스 설정
    series_list = []
    valid_codes = []
    
    for code in codes:
        df = fetch_price_history(code, start, end)
        if df.empty or len(df) < 10:
            continue
            
        # 날짜 컬럼 처리 ('date'가 있다고 가정, 없으면 인덱스 확인 필요)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # 수익률 계산
        pct_series = df['close'].pct_change()
        pct_series.name = code
        series_list.append(pct_series)
        valid_codes.append(code)

    if len(valid_codes) < 2:
        return jsonify({"error": "유효한 데이터 부족"}), 400

    # 2. [중요] 날짜 기준 교집합(Inner Join)으로 데이터 정렬
    returns_df = pd.concat(series_list, axis=1, join='inner').dropna()

    if len(returns_df) < 10:
        return jsonify({"error": "날짜가 겹치는 데이터가 너무 적습니다."}), 400

    try:
        # 3. 효율적 프론티어 곡선 계산 (기존 로직)
        risks, returns, weights = compute_efficient_frontier(returns_df, n_points=40)
        
        frontier_points = []
        for r, ret, w in zip(risks, returns, weights):
            if np.isnan(r) or np.isnan(ret): continue
            frontier_points.append({
                "risk": r, 
                "return": ret, 
                "weights": w
            })

        # 4. [추가] 4가지 핵심 포트폴리오 마커 계산
        markers = compute_frontier_markers(returns_df)

        # 5. 결과 반환
        return jsonify({
            "frontier": frontier_points, # 곡선 그리는 점들
            "markers": markers           # 그래프 위에 찍을 특수 점들 (MinVar, MaxSharpe 등)
        })
        
    except Exception as e:
        print(f"Frontier API Error: {e}")
        return jsonify({"error": str(e)}), 500