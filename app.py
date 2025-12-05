"""
VC Term Sheet Analyzer v2.1 - 한국 자산운용사 최적화 버전
인프라프론티어자산운용(주)

핵심 기능:
1. Term Sheet 조건 입력 (Series A~F)
2. Exit Diagram (Payoff Schedule) 시각화
3. RVPS 기반 전환순서 자동 계산
4. Random Expiration (RE) Option Pricing
5. GP/LP 수익 분배 시뮬레이션
6. Partial Valuation 공식 도출

참고자료:
- Metrick & Yasuda, "Venture Capital and the Finance of Innovation"
- vcvtools.com/auto.php
- 강의자료: Ch9 & 14 Preferred Stock, Ch15 Late Round Investment
"""

import streamlit as st

# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="VC Term Sheet Analyzer | 인프라프론티어",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

# =============================================================================
# 수학 함수 (scipy 없이 직접 구현)
# =============================================================================
def norm_cdf(x):
    """표준정규분포 누적분포함수"""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes 콜옵션 가치"""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, S - K)
    if K <= 0:
        return S
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(0, S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))

def re_option_call(S: float, K: float, H: float, r: float, sigma: float) -> float:
    """Random Expiration Option (VC 투자에 적합한 옵션 모델)"""
    if H <= 0:
        return max(0, S - K)
    total = 0
    for i in range(1, 21):
        t = i * H / 20
        prob = (1 / H) * math.exp(-t / H) * (H / 20)
        total += prob * black_scholes_call(S, K, t, r, sigma)
    return total * H

# =============================================================================
# CSS 스타일 (다크 글래스모피즘)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-card: rgba(20, 20, 30, 0.9);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(99, 102, 241, 0.5);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-accent);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-weight: 900;
        font-size: 1.8rem;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
    }
    .main-header p {
        color: var(--text-secondary);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s;
    }
    .metric-card:hover {
        border-color: var(--accent-primary);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .metric-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }
    
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .formula-box {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #a5b4fc;
        margin: 0.8rem 0;
        overflow-x: auto;
    }
    
    .conversion-order-box {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #6ee7b7;
        font-weight: 600;
    }
    
    .series-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .series-a { background: #6366f1; color: white; }
    .series-b { background: #8b5cf6; color: white; }
    .series-c { background: #a855f7; color: white; }
    .series-d { background: #d946ef; color: white; }
    .series-e { background: #ec4899; color: white; }
    .series-f { background: #f43f5e; color: white; }
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #93c5fd;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #fcd34d;
    }
    
    .section-title {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-accent);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
    }
    
    /* 테이블 스타일 */
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .result-table th {
        background: rgba(99, 102, 241, 0.2);
        color: var(--text-primary);
        padding: 0.8rem;
        text-align: center;
        font-weight: 600;
        border-bottom: 2px solid var(--border-accent);
    }
    .result-table td {
        padding: 0.7rem;
        text-align: center;
        border-bottom: 1px solid var(--border-subtle);
        color: var(--text-secondary);
    }
    .result-table tr:hover td {
        background: rgba(99, 102, 241, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 데이터 클래스
# =============================================================================
@dataclass
class RoundInput:
    """투자 라운드 입력"""
    name: str
    active: bool = False
    security_type: str = "RCPS"  # RCPS, CPS, BW 등
    investment: float = 0  # 투자금액 (억원)
    shares: float = 0  # 주식 수 (만주)
    liquidation_pref: float = 1.0  # 청산우선권 배수
    
    @property
    def redemption_value(self) -> float:
        """상환가치 = 투자금액 × 청산우선권"""
        return self.investment * self.liquidation_pref
    
    @property
    def rvps(self) -> float:
        """주당상환가치 (RVPS) = RV / 주식수"""
        if self.shares > 0:
            return self.redemption_value / self.shares
        return float('inf')

@dataclass
class FundInput:
    """펀드 정보"""
    committed_capital: float = 500  # 약정총액 (억원)
    management_fee_rate: float = 2.0  # 관리보수율 (%)
    carried_interest: float = 20  # 성과보수율 (%)
    hurdle_rate: float = 8.0  # 허들레이트 (%)

@dataclass
class GlobalInput:
    """글로벌 설정"""
    founders_shares: float = 1000  # 창업자 주식 (만주)
    current_valuation: float = 100  # 현재 기업가치 (억원)
    exit_valuation: float = 500  # 예상 Exit 가치 (억원)
    volatility: float = 80  # 변동성 (%)
    risk_free_rate: float = 3.5  # 무위험이자율 (%)
    holding_period: float = 5  # 예상 보유기간 (년)

# =============================================================================
# 핵심 계산 함수
# =============================================================================
def get_conversion_order(rounds: List[RoundInput]) -> List[Tuple[str, float]]:
    """RVPS 기준 전환 순서 계산 (낮은 순)"""
    active = [(r.name, r.rvps) for r in rounds if r.active and r.shares > 0]
    return sorted(active, key=lambda x: x[1])

def calculate_conversion_points(rounds: List[RoundInput], founders_shares: float) -> Dict:
    """각 시리즈의 전환포인트 계산"""
    order = get_conversion_order(rounds)
    results = {}
    
    converted_shares = founders_shares
    remaining_rv = sum(r.redemption_value for r in rounds if r.active)
    
    for name, rvps in order:
        r = next(r for r in rounds if r.name == name)
        total_shares_if_convert = converted_shares + r.shares
        ownership = r.shares / total_shares_if_convert
        prior_rv = remaining_rv - r.redemption_value
        
        # 전환포인트: 전환가치 > 상환가치가 되는 기업가치
        conversion_point = r.redemption_value / ownership + prior_rv
        
        results[name] = {
            'rvps': rvps,
            'rv': r.redemption_value,
            'shares': r.shares,
            'conversion_point': conversion_point,
            'ownership_pct': ownership * 100,
            'order': len(results) + 1
        }
        
        converted_shares += r.shares
        remaining_rv -= r.redemption_value
    
    return results

def calculate_exit_payoffs(exit_value: float, rounds: List[RoundInput], founders_shares: float) -> Dict:
    """특정 Exit 가치에서의 수령액 계산"""
    cp_data = calculate_conversion_points(rounds, founders_shares)
    order = get_conversion_order(rounds)
    
    payoffs = {}
    remaining = exit_value
    converted = set()
    
    # 전환 여부 판단
    for name, _ in order:
        if name in cp_data and exit_value >= cp_data[name]['conversion_point']:
            converted.add(name)
    
    # 상환 (역순: 후순위부터)
    for name, _ in reversed(order):
        if name in converted:
            continue
        r = next(r for r in rounds if r.name == name)
        payout = min(r.redemption_value, remaining)
        payoffs[name] = {'상환': payout, '전환': 0, '합계': payout}
        remaining = max(0, remaining - payout)
    
    # 전환 (지분 배분)
    if remaining > 0:
        total_shares = founders_shares + sum(
            next(r.shares for r in rounds if r.name == n) for n in converted
        )
        
        # 창업자
        founder_payout = (founders_shares / total_shares) * remaining
        payoffs['창업자'] = {'상환': 0, '전환': founder_payout, '합계': founder_payout}
        
        # 전환한 투자자
        for name in converted:
            r = next(r for r in rounds if r.name == name)
            payout = (r.shares / total_shares) * remaining
            payoffs[name] = {'상환': 0, '전환': payout, '합계': payout}
    else:
        payoffs['창업자'] = {'상환': 0, '전환': 0, '합계': 0}
    
    return payoffs

def calculate_partial_valuation(r: RoundInput, rounds: List[RoundInput], 
                                founders_shares: float, g: GlobalInput, use_re: bool = True) -> float:
    """Partial Valuation 계산 (옵션 모델)"""
    cp_data = calculate_conversion_points(rounds, founders_shares)
    
    if r.name not in cp_data:
        return 0
    
    V = g.current_valuation
    rf = g.risk_free_rate / 100
    sigma = g.volatility / 100
    H = g.holding_period
    
    opt_func = re_option_call if use_re else black_scholes_call
    
    data = cp_data[r.name]
    order = get_conversion_order(rounds)
    
    # 선순위 RV 합계
    prior_rv = 0
    for name, _ in order:
        if name == r.name:
            break
        prior_rv += cp_data[name]['rv']
    
    rv = data['rv']
    cp = data['conversion_point']
    ownership = data['ownership_pct'] / 100
    
    # Partial Valuation = C(prior_rv) - C(prior_rv + rv) + ownership × C(cp)
    p1 = opt_func(V, prior_rv, H, rf, sigma) if prior_rv > 0 else V
    p2 = opt_func(V, prior_rv + rv, H, rf, sigma)
    p3 = ownership * opt_func(V, cp, H, rf, sigma)
    
    return max(0, p1 - p2 + p3)

def calculate_lp_cost(fund: FundInput, investment: float) -> float:
    """LP Cost 계산"""
    # 총 관리보수 = 약정총액 × 관리보수율 × 10년 (가정)
    lifetime_fees = fund.committed_capital * (fund.management_fee_rate / 100) * 10
    investable = fund.committed_capital - lifetime_fees
    if investable > 0:
        return (fund.committed_capital / investable) * investment
    return investment

def calculate_gp_lp_split(partial_val: float, fund: FundInput, investment: float) -> Dict:
    """GP/LP 분배 계산"""
    lp_cost = calculate_lp_cost(fund, investment)
    
    # 수익 계산
    profit = max(0, partial_val - investment)
    
    # 허들 적용
    hurdle_amount = investment * (fund.hurdle_rate / 100) * 5  # 5년 가정
    
    if profit <= hurdle_amount:
        gp_carry = 0
    else:
        excess = profit - hurdle_amount
        gp_carry = excess * (fund.carried_interest / 100)
    
    lp_val = partial_val - gp_carry
    
    return {
        'lp_cost': lp_cost,
        'partial_val': partial_val,
        'profit': profit,
        'hurdle': hurdle_amount,
        'gp_carry': gp_carry,
        'lp_valuation': lp_val,
        'lp_return_pct': ((lp_val - lp_cost) / lp_cost * 100) if lp_cost > 0 else 0
    }

# =============================================================================
# 시각화 함수
# =============================================================================
def create_exit_diagram(rounds: List[RoundInput],
                        founders_shares: float,
                        max_exit: float = None) -> go.Figure:
    """Exit Diagram (Composite)"""

    cp_data = calculate_conversion_points(rounds, founders_shares)

    # 전환포인트가 없으면 빈 Figure 반환
    if not cp_data:
        return go.Figure()

    # max_exit 자동 설정
    if max_exit is None:
        finite_cps = [
            d["conversion_point"]
            for d in cp_data.values()
            if d.get("conversion_point") is not None and math.isfinite(d["conversion_point"])
        ]
        if finite_cps:
            max_cp = max(finite_cps)
            max_exit = max_cp * 1.5
        else:
            max_exit = 1000  # fallback

    exit_vals = np.linspace(0, max_exit, 200)

    # 이해관계자 리스트
    parties = ["창업자"] + [r.name for r in rounds if r.active]
    payoff_data = {p: [] for p in parties}

    # 각 Exit 가치마다 수령액 계산
    for ev in exit_vals:
        payoffs = calculate_exit_payoffs(ev, rounds, founders_shares)
        for p in parties:
            payoff_data[p].append(payoffs.get(p, {}).get("합계", 0))

    colors = {
        "창업자": "#10b981",
        "Series A": "#6366f1",
        "Series B": "#f97316",
        "Series C": "#22c55e",
        "Series D": "#d946ef",
        "Series E": "#ec4899",
        "Series F": "#6b7280",
    }

    fig = go.Figure()

    # 각 이해관계자 라인 추가
    for p in parties:
        fig.add_trace(
            go.Scatter(
                x=exit_vals,
                y=payoff_data[p],
                name=p,
                mode="lines",
                line=dict(width=3, color=colors.get(p, "#64748b")),
                hovertemplate=(
                    f"<b>{p}</b><br>"
                    "Exit: %{x:.1f}억<br>"
                    "수령액: %{y:.2f}억<extra></extra>"
                ),
            )
        )

    # 전환포인트 수직선 및 라벨
    for name, data in cp_data.items():
        cp = data.get("conversion_point")
        if cp is None or not math.isfinite(cp):
            continue

        fig.add_vline(
            x=cp,
            line_dash="dash",
            line_color=colors.get(name, "#64748b"),
        )
        fig.add_annotation(
            x=cp,
            y=0,
            yref="paper",
            yanchor="bottom",
            showarrow=False,
            text=f"{name} CP",
            font=dict(size=10, color=colors.get(name, "#64748b")),
        )

    fig.update_layout(
        title=dict(
            text="Exit Diagram (Composite)",
            font=dict(size=16, color="#f8fafc"),
        ),
        xaxis=dict(
            title=dict(text="Exit 가치 (억원)", font=dict(color="#94a3b8")),
            tickfont=dict(color="#64748b"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title=dict(text="수령액 (억원)", font=dict(color="#94a3b8")),
            tickfont=dict(color="#64748b"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(20,20,30,0.8)",
            font=dict(color="#f8fafc"),
        ),
        hovermode="x unified",
        height=450,
    )

    return fig


def create_series_diagrams(rounds: List[RoundInput], founders_shares: float, max_exit: float = None) -> go.Figure:
    """개별 Series Exit Diagram"""
    active = [r for r in rounds if r.active]
    if not active:
        return go.Figure()
    
    cp_data = calculate_conversion_points(rounds, founders_shares)
    
    if max_exit is None:
        max_cp = max(d['conversion_point'] for d in cp_data.values())
        max_exit = max_cp * 1.5
    
    exit_vals = np.linspace(0, max_exit, 200)
    
    n_plots = min(len(active) + 1, 4)
    titles = ['창업자'] + [r.name for r in active[:3]]
    
    fig = make_subplots(rows=1, cols=n_plots, subplot_titles=titles, horizontal_spacing=0.08)
    
    colors = {'창업자': '#10b981', 'Series A': '#6366f1', 'Series B': '#f97316', 'Series C': '#22c55e'}
    parties = ['창업자'] + [r.name for r in active]
    
    for idx, party in enumerate(parties[:n_plots]):
        payoffs = []
        for ev in exit_vals:
            p = calculate_exit_payoffs(ev, rounds, founders_shares)
            payoffs.append(p.get(party, {}).get('합계', 0))
        
        fig.add_trace(
            go.Scatter(x=exit_vals, y=payoffs, line=dict(width=2, color=colors.get(party, '#64748b')), showlegend=False),
            row=1, col=idx+1
        )
    
    fig.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', title_text='Exit (억원)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', title_text='수령액')
    
    return fig

def create_waterfall_chart(gp_lp_data: Dict, series_name: str) -> go.Figure:
    """GP/LP 분배 워터폴 차트"""
    fig = go.Figure(go.Waterfall(
        name="분배 흐름",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=["투자원금", "수익", "허들 공제", "GP Carry", "LP 수령액"],
        y=[
            gp_lp_data['lp_cost'],
            gp_lp_data['profit'],
            -gp_lp_data['hurdle'] if gp_lp_data['hurdle'] > 0 else 0,
            -gp_lp_data['gp_carry'],
            0
        ],
        connector={"line": {"color": "rgba(99,102,241,0.5)"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#6366f1"}}
    ))
    
    fig.update_layout(
        title=f"{series_name} GP/LP 분배 워터폴",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=350
    )
    
    return fig

# =============================================================================
# 메인 앱
# =============================================================================
def main():
    # 세션 상태 초기화
    if 'rounds' not in st.session_state:
        st.session_state.rounds = [
            RoundInput(name="Series A"),
            RoundInput(name="Series B"),
            RoundInput(name="Series C"),
            RoundInput(name="Series D"),
            RoundInput(name="Series E"),
            RoundInput(name="Series F"),
        ]
    if 'global_input' not in st.session_state:
        st.session_state.global_input = GlobalInput()
    if 'fund_input' not in st.session_state:
        st.session_state.fund_input = FundInput()
    
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📊 VC Term Sheet Analyzer</h1>
        <p>상환전환우선주(RCPS) 조건 분석 | Exit Diagram | GP/LP 수익 시뮬레이션</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # 사이드바
    # ==========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        
        st.markdown("### 👤 창업자 정보")
        st.session_state.global_input.founders_shares = st.number_input(
            "창업자 보통주 (만주)", min_value=1, max_value=100000000,
            value=int(st.session_state.global_input.founders_shares), step=100,
            format="%d"
        )
        
        st.markdown("### 💰 기업가치")
        st.session_state.global_input.current_valuation = st.number_input(
            "현재 기업가치 (억원)", min_value=1.0, max_value=100000.0,
            value=float(st.session_state.global_input.current_valuation), step=10.0
        )
        
        st.session_state.global_input.exit_valuation = st.number_input(
            "예상 Exit 가치 (억원)", min_value=1.0, max_value=100000.0,
            value=float(st.session_state.global_input.exit_valuation), step=50.0
        )
        
        st.markdown("### 📈 옵션 파라미터")
        st.caption("📖 Base-Case Assumptions (Cochrane, 2005)")
        
        st.session_state.global_input.volatility = st.slider(
            "변동성 (%)", 20, 150, int(st.session_state.global_input.volatility),
            help="스타트업 평균 변동성: 80~90%"
        )
        
        st.session_state.global_input.risk_free_rate = st.slider(
            "무위험이자율 (%)", 0.0, 10.0, float(st.session_state.global_input.risk_free_rate), 0.5
        )
        
        st.session_state.global_input.holding_period = st.slider(
            "예상 보유기간 (년)", 1, 15, int(st.session_state.global_input.holding_period),
            help="Series A: 5년, B: 4년, C이후: 3년"
        )
        
        st.markdown("---")
        st.markdown("### 🏦 펀드 정보")
        
        st.session_state.fund_input.committed_capital = st.number_input(
            "약정총액 (억원)", min_value=10.0, max_value=10000.0,
            value=float(st.session_state.fund_input.committed_capital), step=50.0
        )
        
        st.session_state.fund_input.management_fee_rate = st.slider(
            "관리보수 (%)", 0.0, 5.0, float(st.session_state.fund_input.management_fee_rate), 0.25
        )
        
        st.session_state.fund_input.carried_interest = st.slider(
            "성과보수 (%)", 0.0, 30.0, float(st.session_state.fund_input.carried_interest), 1.0
        )
        
        st.session_state.fund_input.hurdle_rate = st.slider(
            "허들레이트 (%)", 0.0, 15.0, float(st.session_state.fund_input.hurdle_rate), 0.5
        )
    
    # ==========================================================================
    # 탭 구성
    # ==========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 투자조건 입력", "📊 Exit Diagram", "💼 Valuation 분석", "📖 사용법"
    ])
    
    # =========================================================================
    # TAB 1: 투자조건 입력
    # =========================================================================
    with tab1:
        st.markdown('<div class="section-title">📝 EXIT DIAGRAM INPUTS</div>', unsafe_allow_html=True)
        st.caption("vcvtools.com 방식의 Term Sheet 입력")
        
        # 라운드 활성화 체크박스
        cols = st.columns(6)
        for idx, r in enumerate(st.session_state.rounds):
            with cols[idx]:
                badge_class = r.name.lower().replace(" ", "-")
                st.markdown(f"<span class='series-badge {badge_class}'>{r.name}</span>", unsafe_allow_html=True)
                r.active = st.checkbox("활성", value=r.active, key=f"active_{r.name}", label_visibility="collapsed")
        
        st.markdown("---")
        
        # 활성 라운드 상세 입력
        active_rounds = [r for r in st.session_state.rounds if r.active]
        
        if active_rounds:
            st.markdown("#### 라운드별 상세 조건")
            
            # 입력 폼
            input_cols = st.columns(len(active_rounds))
            
            for idx, r in enumerate(active_rounds):
                with input_cols[idx]:
                    st.markdown(f"**{r.name}**")
                    
                    r.security_type = st.selectbox(
                        "증권유형", ["RCPS", "CPS", "BW", "CB"],
                        key=f"type_{r.name}",
                        help="RCPS: 상환전환우선주, CPS: 전환우선주"
                    )
                    
                    r.investment = st.number_input(
                        "투자금액 (억원)", min_value=0.0, max_value=10000.0,
                        value=float(r.investment), step=1.0, key=f"inv_{r.name}"
                    )
                    
                    r.shares = st.number_input(
                        "주식수 (만주)", min_value=0.0, max_value=100000.0,
                        value=float(r.shares), step=10.0, key=f"shares_{r.name}"
                    )
                    
                    r.liquidation_pref = st.selectbox(
                        "청산우선권", [1.0, 1.5, 2.0, 2.5, 3.0],
                        index=0, key=f"lp_{r.name}",
                        help="상환 시 투자금액의 배수"
                    )
            
            st.markdown("---")
            
            # RVPS 및 전환순서
            valid_rounds = [r for r in active_rounds if r.shares > 0]
            
            if valid_rounds:
                st.markdown('<div class="section-title">📋 전환순서 (Conversion Order)</div>', unsafe_allow_html=True)
                st.caption("📖 강의자료: Conversion-Order Shortcut - RVPS가 낮을수록 먼저 전환")
                
                order = get_conversion_order(st.session_state.rounds)
                
                # RVPS 테이블
                rvps_html = """
                <table class="result-table">
                <tr><th>Series</th><th>투자금액</th><th>주식수 (만주)</th><th>청산배수</th><th>상환가치 (RV)</th><th>RVPS</th></tr>
                """
                for name, rvps in order:
                    r = next(r for r in st.session_state.rounds if r.name == name)
                    rvps_html += f"""
                    <tr>
                        <td><span class="series-badge {name.lower().replace(' ','-')}">{name}</span></td>
                        <td>{r.investment:.1f}억</td>
                        <td>{r.shares:.0f}</td>
                        <td>{r.liquidation_pref}x</td>
                        <td>{r.redemption_value:.1f}억</td>
                        <td><strong>{rvps:.4f}</strong></td>
                    </tr>
                    """
                rvps_html += "</table>"
                st.markdown(rvps_html, unsafe_allow_html=True)
                
                # 전환순서 표시
                order_badges = " → ".join([f"<span class='series-badge {n.lower().replace(' ','-')}'>{n}</span>" for n, _ in order])
                st.markdown(f"""
                <div class="conversion-order-box">
                    <strong>전환순서:</strong> {order_badges}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    💡 <strong>해석:</strong> RVPS가 낮다 = 주당 상환받을 금액이 적다 = 전환해서 지분을 받는 것이 더 빨리 유리해짐
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👆 위에서 분석할 Series를 선택하세요.")
    
    # =========================================================================
    # TAB 2: Exit Diagram
    # =========================================================================
    with tab2:
        st.markdown('<div class="section-title">📊 Exit Diagram</div>', unsafe_allow_html=True)
        st.caption("📖 강의자료: 전환 또는 상환 결정 (p.5), Exit Valuation of CP (p.6)")
        
        valid_rounds = [r for r in st.session_state.rounds if r.active and r.shares > 0]
        
        if not valid_rounds:
            st.warning("📝 투자조건 입력 탭에서 라운드 정보를 입력하세요.")
        else:
            cp_data = calculate_conversion_points(
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            
            # 전환포인트 메트릭
            st.markdown("#### 전환포인트 (Conversion Points)")
            
            cp_cols = st.columns(len(cp_data))
            for idx, (name, data) in enumerate(cp_data.items()):
                with cp_cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value">{data['conversion_point']:.1f}억</div>
                        <div class="metric-sub">지분율: {data['ownership_pct']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 개별 Exit Diagram
            st.markdown("#### Series Diagrams")
            fig_series = create_series_diagrams(
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            st.plotly_chart(fig_series, width="stretch")
            
            # Composite Diagram
            st.markdown("#### Composite Diagram")
            fig_composite = create_exit_diagram(
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            st.plotly_chart(fig_composite, width="stretch")
            
            # 특정 Exit Value 분석
            st.markdown("---")
            st.markdown("#### 특정 Exit 가치에서의 분배")
            
            max_cp = max(d['conversion_point'] for d in cp_data.values())
            exit_val = st.slider(
                "Exit 가치 (억원)",
                min_value=0.0,
                max_value=float(max_cp * 2),
                value=float(st.session_state.global_input.exit_valuation)
            )
            
            payoffs = calculate_exit_payoffs(
                exit_val,
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            
            # 분배 결과 테이블
            payoff_html = """
            <table class="result-table">
            <tr><th>이해관계자</th><th>상환액</th><th>전환액</th><th>합계</th><th>비율</th></tr>
            """
            for party, data in payoffs.items():
                pct = (data['합계'] / exit_val * 100) if exit_val > 0 else 0
                payoff_html += f"""
                <tr>
                    <td><strong>{party}</strong></td>
                    <td>{data['상환']:.2f}억</td>
                    <td>{data['전환']:.2f}억</td>
                    <td><strong>{data['합계']:.2f}억</strong></td>
                    <td>{pct:.1f}%</td>
                </tr>
                """
            payoff_html += "</table>"
            st.markdown(payoff_html, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 3: Valuation 분석
    # =========================================================================
    with tab3:
        st.markdown('<div class="section-title">💼 AUTO OUTPUTS - Valuation 분석</div>', unsafe_allow_html=True)
        st.caption("📖 강의자료: Option Pricing Model, GP/LP 분배")
        
        valid_rounds = [r for r in st.session_state.rounds if r.active and r.shares > 0]
        
        if not valid_rounds:
            st.warning("📝 투자조건 입력 탭에서 라운드 정보를 입력하세요.")
        else:
            # Partial Valuation 결과
            st.markdown("#### Partial Valuation & GP/LP 분배")
            
            results = []
            for r in valid_rounds:
                partial_val = calculate_partial_valuation(
                    r, st.session_state.rounds,
                    st.session_state.global_input.founders_shares,
                    st.session_state.global_input,
                    use_re=True
                )
                
                gp_lp = calculate_gp_lp_split(
                    partial_val,
                    st.session_state.fund_input,
                    r.investment
                )
                
                results.append({
                    'series': r.name,
                    'investment': r.investment,
                    **gp_lp
                })
            
            # 결과 테이블
            result_html = """
            <table class="result-table">
            <tr><th>Series</th><th>투자금액</th><th>LP Cost</th><th>Partial Val</th><th>GP Carry</th><th>LP Valuation</th><th>LP 수익률</th></tr>
            """
            for res in results:
                return_color = '#10b981' if res['lp_return_pct'] >= 0 else '#ef4444'
                result_html += f"""
                <tr>
                    <td><span class="series-badge {res['series'].lower().replace(' ','-')}">{res['series']}</span></td>
                    <td>{res['investment']:.1f}억</td>
                    <td>{res['lp_cost']:.2f}억</td>
                    <td><strong>{res['partial_val']:.2f}억</strong></td>
                    <td>{res['gp_carry']:.2f}억</td>
                    <td><strong>{res['lp_valuation']:.2f}억</strong></td>
                    <td style="color:{return_color}"><strong>{res['lp_return_pct']:.1f}%</strong></td>
                </tr>
                """
            result_html += "</table>"
            st.markdown(result_html, unsafe_allow_html=True)
            
            # 워터폴 차트
            st.markdown("---")
            st.markdown("#### GP/LP 분배 워터폴")
            
            selected_series = st.selectbox(
                "Series 선택",
                [r['series'] for r in results]
            )
            
            selected_data = next(r for r in results if r['series'] == selected_series)
            fig_waterfall = create_waterfall_chart(selected_data, selected_series)
            st.plotly_chart(fig_waterfall, width="stretch")
            
            # Breakeven 계산
            st.markdown("---")
            st.markdown("#### Implied-post Valuation (Breakeven)")
            
            if st.button("🎯 Breakeven 계산", type="primary"):
                target = valid_rounds[-1]
                lp_cost = calculate_lp_cost(st.session_state.fund_input, target.investment)
                
                # Binary search
                low, high = 10, 10000
                for _ in range(50):
                    mid = (low + high) / 2
                    test_g = GlobalInput(
                        founders_shares=st.session_state.global_input.founders_shares,
                        current_valuation=mid,
                        exit_valuation=mid,
                        volatility=st.session_state.global_input.volatility,
                        risk_free_rate=st.session_state.global_input.risk_free_rate,
                        holding_period=st.session_state.global_input.holding_period
                    )
                    pv = calculate_partial_valuation(target, st.session_state.rounds,
                                                     test_g.founders_shares, test_g)
                    gp_lp = calculate_gp_lp_split(pv, st.session_state.fund_input, target.investment)
                    
                    if gp_lp['lp_valuation'] < lp_cost:
                        low = mid
                    else:
                        high = mid
                
                st.success(f"**{target.name} Implied-post Valuation:** {mid:.2f}억원")
                st.caption(f"이 기업가치에서 LP Cost ({lp_cost:.2f}억) = LP Valuation")
    
    # =========================================================================
    # TAB 4: 사용법
    # =========================================================================
    with tab4:
        st.markdown('<div class="section-title">📖 사용 가이드</div>', unsafe_allow_html=True)
        
        st.markdown("""
        #### 🎯 도구 개요
        
        이 도구는 **VC 투자의 Term Sheet 조건**을 분석하고, **Exit 시나리오별 수익 분배**를 시뮬레이션합니다.
        
        ---
        
        #### 📊 주요 개념
        """)
        
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#6366f1;">1. RVPS (Redemption Value Per Share)</h4>
        <div class="formula-box">RVPS = 상환가치(RV) / 전환 시 받을 주식수</div>
        <p style="color:#94a3b8;">• RVPS가 낮을수록 먼저 전환 (전환이 유리한 시점이 빨리 옴)<br>• Conversion Order 결정의 핵심 지표</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#8b5cf6;">2. 전환포인트 (Conversion Point)</h4>
        <div class="formula-box">전환 조건: 지분율 × (기업가치 - 선순위 RV) > 나의 RV</div>
        <p style="color:#94a3b8;">• 이 조건을 만족하는 최소 기업가치<br>• 이 가치 이상이면 상환보다 전환이 유리</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#a855f7;">3. Partial Valuation</h4>
        <div class="formula-box">CP 가치 = V - C(K₁) + α×C(K₂) - β×C(K₃) ...</div>
        <p style="color:#94a3b8;">• V: 기업가치, C(K): Strike K인 콜옵션 가치<br>• Random Expiration (RE) Option 모델로 계산<br>• 각 시리즈의 실제 경제적 가치</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#d946ef;">4. LP/GP 분배</h4>
        <div class="formula-box">
        LP Cost = (약정총액 / 투자가능액) × 투자금액<br>
        GP Carry = (수익 - 허들) × 성과보수율<br>
        LP Valuation = Partial Valuation - GP Carry
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ---
        
        #### 🔧 옵션 파라미터 기본값
        
        | 파라미터 | 권장값 | 출처 |
        |---------|--------|------|
        | 변동성 | 80~90% | Cochrane (2005) |
        | 무위험이자율 | 3~5% | 한국 국고채 기준 |
        | 보유기간 | Series A: 5년, B: 4년, C+: 3년 | 교재 기본값 |
        
        ---
        
        #### 📚 참고 자료
        
        - **원본 도구**: [vcvtools.com](http://vcvtools.com/)
        - **교재**: Metrick & Yasuda, *Venture Capital and the Finance of Innovation*
        - **강의**: Ch9 & 14 Preferred Stock, Ch15 Late Round Investment
        """)
        
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <h4 style="color:#6366f1;">🏢 인프라프론티어자산운용(주)</h4>
            <p style="color:#94a3b8;">VC Term Sheet Analyzer v2.1</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
