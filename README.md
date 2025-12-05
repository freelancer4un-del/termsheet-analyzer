# 🚀 VC Term Sheet Analyzer

한국 VC 시장을 위한 Term Sheet 분석 도구

## 📋 개요

[vcvtools.com/auto.php](http://vcvtools.com/auto.php)를 벤치마킹하여 한국 VC 시장에 맞게 재구현한 Term Sheet 분석 도구입니다.

## ✨ 주요 기능

### 1. Term Sheet 입력
- **Series A~F** 최대 6개 라운드 분석
- 증권 유형: CP, RP, PCP, PCPC
- 청산우선권, 참가권, 희석방지조항 설정

### 2. Exit Diagram
- Exit 가치별 Payoff Schedule 시각화
- 창업자, 투자자별 수령액 분석
- 청산우선권, 참가권, 전환권 행사 시뮬레이션

### 3. GP/LP 분석
- VC 펀드 GP/LP 수익 분배
- 관리보수, 성과보수(Carry), 허들레이트 반영
- LP 순수익률 계산

## 🎯 대상 사용자

- **스타트업 창업자**: Term Sheet 협상 시뮬레이션
- **VC 심사역**: 투자 조건별 수익 분석
- **LP 담당자**: 펀드 투자 예상 수익률 검토
- **교육용**: VC 투자 구조 이해

## 🛠️ 설치 및 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# 실행
streamlit run app.py
```

## 📊 용어 설명

| 용어 | 설명 |
|------|------|
| CP | 전환우선주 (Convertible Preferred) |
| RP | 상환우선주 (Redeemable Preferred) |
| PCP | 참가적 전환우선주 (Participating CP) |
| Liquidation Preference | 청산우선권 |
| Carry | GP 성과보수 |
| Hurdle Rate | LP 최소 보장 수익률 |

## 📚 참고

- 원본: [vcvtools.com](http://vcvtools.com/)
- 교재: "Venture Capital and the Finance of Innovation" (Metrick & Yasuda)

## 📄 라이선스

인프라프론티어자산운용(주) 내부 사용 목적

---

🏢 인프라프론티어자산운용(주) | VC Term Sheet Analyzer v1.0
