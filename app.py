"""
VC Term Sheet Analyzer v2.0 - 한국 VC 시장용
벤치마킹: vcvtools.com/auto.php
강의 내용 반영: Ch9 & 14 Preferred Stock, Ch15 Late Round Investment

핵심 기능:
1. Term Sheet 입력 → 지분 분배 계산
2. Series A~F 라운드별 분석
3. GP/LP 수익 분배 시뮬레이션
4. Exit Diagram (Payoff Schedule) 시각화
5. Random Expiration (RE) Option Pricing
6. RVPS 기반 Conversion Order 계산
7. Partial Valuation 공식 도출
"""

import streamlit as st

# =============================================================================
# 페이지 설정 (반드시 첫 번째!)
# =============================================================================
st.set_page_config(
    page_title="🚀 VC Term Sheet Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math

# =============================================================================
# Black-Scholes & RE Option (scipy 없이 직접 구현)
# =============================================================================
def norm_cdf(x):
    """표준정규분포 CDF (scipy 없이 구현)"""
    # Abramowitz and Stegun approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return 0.5 * (1.0 + sign * y)

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European Call Option 가치
    📌 강의자료: Base-Case Option Pricing Assumptions (p.12)
    
    S: 기초자산 가치 (Total Valuation)
    K: 행사가격 (Strike = Conversion Point)
    T: 만기 (Expected Holding Period)
    r: 무위험이자율 (Risk Free Rate)
    sigma: 변동성 (Volatility)
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, S - K)
    
    if K <= 0:
        return S
    
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_value = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return max(0, call_value)

def random_expiration_call(S: float, K: float, H: float, r: float, sigma: float, 
                           num_periods: int = 20) -> float:
    """
    Random Expiration (RE) Option 가치
    📌 강의자료: Random Expiration (RE) Options & CP (p.9)
    
    RE Option = 만기 도래 확률 * European Call의 적분
    실제로는 여러 만기의 European Call의 가중평균으로 근사
    
    H: Expected Holding Period (기대 보유기간)
    """
    if H <= 0:
        return max(0, S - K)
    
    # 만기를 여러 기간으로 분할하여 가중평균
    total_value = 0
    dt = H / num_periods
    
    for i in range(1, num_periods + 1):
        t = i * dt
        # 지수분포 가정: 만기 도래 확률
        prob = (1 / H) * math.exp(-t / H) * dt
        call_value = black_scholes_call(S, K, t, r, sigma)
        total_value += prob * call_value
    
    # 정규화
    total_value = total_value * H
    
    return total_value

# =============================================================================
# CSS 스타일
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(20, 20, 30, 0.8);
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
        --accent-danger: #ef4444;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-accent);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-weight: 800;
        font-size: 2.2rem;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    .main-header p {
        color: var(--text-secondary);
        margin: 0;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(20, 20, 30, 0.9) 0%, rgba(15, 15, 25, 0.9) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .metric-sub {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
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
        border-radius: 12px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #a5b4fc;
        margin: 1rem 0;
    }
    
    .conversion-order {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #6ee7b7;
        font-weight: 600;
    }
    
    .series-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .series-a { background: #6366f1; color: white; }
    .series-b { background: #8b5cf6; color: white; }
    .series-c { background: #a855f7; color: white; }
    .series-d { background: #d946ef; color: white; }
    .series-e { background: #ec4899; color: white; }
    .series-f { background: #f43f5e; color: white; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 데이터 클래스 정의
# =============================================================================
@dataclass
class RoundInput:
    """투자 라운드 입력 데이터"""
    name: str
    active: bool = False
    security_type: str = "CP"  # CP, RP, PCP
    investment: float = 0  # 투자금액 (백만원 또는 M$)
    shares: float = 0  # 주식 수 (백만주)
    liquidation_pref: float = 1.0  # 청산우선권 배수
    participation_cap: float = 0  # 참가권 상한
    
    @property
    def redemption_value(self) -> float:
        """상환가치 (RV) = 투자금액 × 청산우선권 배수"""
        return self.investment * self.liquidation_pref
    
    @property
    def rvps(self) -> float:
        """
        Redemption Value Per Share
        📌 강의자료: Conversion-Order Shortcut (p.28)
        RVPS = RV / 전환시 받을 주식수
        """
        if self.shares > 0:
            return self.redemption_value / self.shares
        return float('inf')

@dataclass
class FundInput:
    """펀드 입력 데이터"""
    committed_capital: float = 100  # 약정총액
    lifetime_fees: float = 20  # 총 관리보수
    gp_percent: float = 20  # GP% (Carried Interest)

@dataclass
class GlobalInput:
    """글로벌 입력 데이터"""
    founders_shares: float = 10  # 창업자 주식 수 (백만주)
    total_valuation: float = 100  # 총 기업가치
    volatility: float = 90  # 변동성 (%) - 강의자료 기본값
    risk_free_rate: float = 5  # 무위험이자율 (%) - 강의자료 기본값
    holding_period: float = 5  # 기대 보유기간 (년) - Series A 기본값

# =============================================================================
# 핵심 계산 함수들
# =============================================================================
def get_conversion_order(rounds: List[RoundInput]) -> List[Tuple[str, float]]:
    """
    전환 순서 계산 (RVPS 기준)
    📌 강의자료: Conversion-Order Shortcut (p.28)
    - RVPS가 낮은 순서대로 전환
    """
    active_rounds = [(r.name, r.rvps) for r in rounds if r.active and r.shares > 0]
    return sorted(active_rounds, key=lambda x: x[1])

def calculate_conversion_points(rounds: List[RoundInput], founders_shares: float) -> Dict:
    """
    각 Series의 전환 포인트(Conversion Point) 계산
    📌 강의자료: Structure: 10M shares CP (p.20)
    
    전환 조건: (지분율) × (W - 선순위 RV) > RV
    → W = RV / 지분율 + 선순위 RV
    """
    conversion_order = get_conversion_order(rounds)
    results = {}
    
    # 현재까지 전환된 주식수 (창업자 포함)
    converted_shares = founders_shares
    # 남은 상환가치 합계
    remaining_rv = sum(r.redemption_value for r in rounds if r.active)
    
    for name, rvps in conversion_order:
        r = next(r for r in rounds if r.name == name)
        
        # 전환 시점의 지분율
        total_shares_if_convert = converted_shares + r.shares
        ownership_if_convert = r.shares / total_shares_if_convert
        
        # 전환 포인트 계산
        # 전환 조건: ownership × (W - remaining_rv + r.rv) > r.rv
        # → W > r.rv / ownership + remaining_rv - r.rv
        prior_rv = remaining_rv - r.redemption_value
        conversion_point = r.redemption_value / ownership_if_convert + prior_rv
        
        results[name] = {
            'rvps': rvps,
            'rv': r.redemption_value,
            'shares': r.shares,
            'conversion_point': conversion_point,
            'ownership_at_conversion': ownership_if_convert * 100,
            'order': len(results) + 1
        }
        
        # 업데이트
        converted_shares += r.shares
        remaining_rv -= r.redemption_value
    
    return results

def calculate_partial_valuation_formula(rounds: List[RoundInput], founders_shares: float) -> Dict:
    """
    Partial Valuation 공식 도출
    📌 강의자료: (2) Partial valuation for Series A (p.32)
    
    각 시리즈의 가치 = 옵션들의 조합
    예: V - C(RV1) + α×C(CP1) - β×C(CP2) - ...
    """
    conversion_data = calculate_conversion_points(rounds, founders_shares)
    conversion_order = get_conversion_order(rounds)
    
    results = {}
    
    for target_name, _ in conversion_order:
        target_data = conversion_data[target_name]
        
        # 옵션 공식 구성
        formula_parts = []
        
        # 기본: 상환가치까지는 채권처럼
        # 첫 번째 옵션: 기업가치에서 자기 RV까지의 권리
        cumulative_rv = 0
        for name, _ in conversion_order:
            if name == target_name:
                break
            cumulative_rv += conversion_data[name]['rv']
        
        # 시작점
        start_rv = cumulative_rv
        target_rv = target_data['rv']
        
        formula_parts.append(f"C({start_rv:.0f})")
        formula_parts.append(f"- C({start_rv + target_rv:.0f})")
        
        # 전환 이후 지분 참여
        total_shares = founders_shares
        for name, _ in conversion_order:
            total_shares += next(r.shares for r in rounds if r.name == name)
            cp = conversion_data[name]['conversion_point']
            
            # 지분율 변화에 따른 옵션 추가
            if name == target_name:
                ownership = conversion_data[name]['shares'] / total_shares
                formula_parts.append(f"+ {ownership:.4f}×C({cp:.0f})")
            elif conversion_order.index((name, conversion_data[name]['rvps'])) > \
                 conversion_order.index((target_name, target_data['rvps'])):
                # 후순위 전환 시 지분율 감소
                prev_total = total_shares - next(r.shares for r in rounds if r.name == name)
                delta_ownership = (conversion_data[target_name]['shares'] / prev_total) - \
                                 (conversion_data[target_name]['shares'] / total_shares)
                if delta_ownership > 0.001:
                    formula_parts.append(f"- {delta_ownership:.4f}×C({cp:.0f})")
        
        results[target_name] = {
            'formula': ' '.join(formula_parts),
            'conversion_point': target_data['conversion_point'],
            'order': target_data['order']
        }
    
    return results

def calculate_lp_cost(fund: FundInput, investment: float) -> float:
    """
    LP Cost 계산
    📌 강의자료: Talltree Fund (p.19)
    LP Cost = (Committed Capital / Investable Capital) × Investment
    Investable Capital = Committed Capital - Lifetime Fees
    """
    investable = fund.committed_capital - fund.lifetime_fees
    if investable > 0:
        return (fund.committed_capital / investable) * investment
    return investment

def calculate_partial_valuation(round_input: RoundInput, 
                                rounds: List[RoundInput],
                                founders_shares: float,
                                global_input: GlobalInput,
                                use_re_option: bool = True) -> float:
    """
    Partial Valuation 계산 (옵션 가치 합산)
    📌 강의자료: 상환전환우선주 Exit Diagram (p.11)
    Value of CP = V - C(K1) + α×C(K2)
    """
    conversion_data = calculate_conversion_points(rounds, founders_shares)
    
    if round_input.name not in conversion_data:
        return 0
    
    V = global_input.total_valuation
    r = global_input.risk_free_rate / 100
    sigma = global_input.volatility / 100
    H = global_input.holding_period
    
    # 옵션 가치 함수 선택
    option_func = random_expiration_call if use_re_option else black_scholes_call
    
    target_data = conversion_data[round_input.name]
    
    # 간단한 근사: V - C(RV 시작점) + 지분율 × C(전환점)
    cumulative_rv = 0
    for name, rvps in get_conversion_order(rounds):
        if name == round_input.name:
            break
        cumulative_rv += conversion_data[name]['rv']
    
    start_rv = cumulative_rv
    target_rv = target_data['rv']
    cp = target_data['conversion_point']
    ownership = target_data['ownership_at_conversion'] / 100
    
    # Partial Valuation = C(start) - C(start + rv) + ownership × C(conversion_point)
    part1 = option_func(V, start_rv, H, r, sigma) if start_rv > 0 else V
    part2 = option_func(V, start_rv + target_rv, H, r, sigma)
    part3 = ownership * option_func(V, cp, H, r, sigma)
    
    partial_val = part1 - part2 + part3
    
    return max(0, partial_val)

def calculate_gp_lp_valuation(partial_val: float, fund: FundInput, investment: float) -> Dict:
    """
    GP/LP Valuation 계산
    📌 강의자료: AUTO OUTPUTS (p.24)
    """
    lp_cost = calculate_lp_cost(fund, investment)
    gp_val = partial_val * (fund.gp_percent / 100)
    lp_val = partial_val - gp_val
    
    return {
        'lp_cost': lp_cost,
        'partial_valuation': partial_val,
        'gp_valuation': gp_val,
        'lp_valuation': lp_val
    }

def calculate_exit_payoffs(exit_value: float, rounds: List[RoundInput], 
                           founders_shares: float) -> Dict:
    """
    특정 Exit 가치에서의 Payoff 계산
    📌 강의자료: Exit Valuation of CP (p.6)
    """
    conversion_data = calculate_conversion_points(rounds, founders_shares)
    conversion_order = get_conversion_order(rounds)
    
    payoffs = {}
    remaining_value = exit_value
    converted_rounds = set()
    
    # 전환 여부 결정
    for name, _ in conversion_order:
        cp = conversion_data[name]['conversion_point']
        if exit_value >= cp:
            converted_rounds.add(name)
    
    # 상환 우선 (역순: 후순위부터)
    for name, _ in reversed(conversion_order):
        if name in converted_rounds:
            continue
        
        r = next(r for r in rounds if r.name == name)
        rv = r.redemption_value
        payout = min(rv, remaining_value)
        payoffs[name] = {'redemption': payout, 'conversion': 0, 'total': payout}
        remaining_value = max(0, remaining_value - payout)
    
    # 전환 (지분 분배)
    if remaining_value > 0:
        total_converted_shares = founders_shares
        for name in converted_rounds:
            r = next(r for r in rounds if r.name == name)
            total_converted_shares += r.shares
        
        # 창업자 몫
        founder_share = (founders_shares / total_converted_shares) * remaining_value
        payoffs['founders'] = {'redemption': 0, 'conversion': founder_share, 'total': founder_share}
        
        # 전환된 투자자 몫
        for name in converted_rounds:
            r = next(r for r in rounds if r.name == name)
            share_payout = (r.shares / total_converted_shares) * remaining_value
            payoffs[name] = {'redemption': 0, 'conversion': share_payout, 'total': share_payout}
    else:
        payoffs['founders'] = {'redemption': 0, 'conversion': 0, 'total': 0}
    
    return payoffs

# =============================================================================
# 시각화 함수들
# =============================================================================
def create_exit_diagram(rounds: List[RoundInput], founders_shares: float, 
                        max_exit: float = None) -> go.Figure:
    """
    Exit Diagram 생성
    📌 강의자료: SERIES DIAGRAMS & COMPOSITE DIAGRAM (p.25)
    """
    if max_exit is None:
        # 최대 전환점의 1.5배
        conversion_data = calculate_conversion_points(rounds, founders_shares)
        max_cp = max([d['conversion_point'] for d in conversion_data.values()], default=100)
        max_exit = max_cp * 1.5
    
    exit_values = np.linspace(0, max_exit, 200)
    
    # 각 이해관계자별 Payoff 계산
    all_parties = ['founders'] + [r.name for r in rounds if r.active]
    payoff_data = {party: [] for party in all_parties}
    
    for ev in exit_values:
        payoffs = calculate_exit_payoffs(ev, rounds, founders_shares)
        for party in all_parties:
            if party in payoffs:
                payoff_data[party].append(payoffs[party]['total'])
            else:
                payoff_data[party].append(0)
    
    # Plotly Figure
    fig = go.Figure()
    
    colors = {
        'founders': '#10b981',
        'Series A': '#6366f1',
        'Series B': '#f97316',
        'Series C': '#22c55e',
        'Series D': '#d946ef',
        'Series E': '#ec4899',
        'Series F': '#6b7280',
    }
    
    for party in all_parties:
        color = colors.get(party, '#64748b')
        display_name = '창업자' if party == 'founders' else party
        fig.add_trace(go.Scatter(
            x=exit_values,
            y=payoff_data[party],
            name=display_name,
            mode='lines',
            line=dict(width=3, color=color),
            hovertemplate=f'<b>{display_name}</b><br>Exit: %{{x:.1f}}<br>Payoff: %{{y:.2f}}<extra></extra>'
        ))
    
    # 전환점 표시
    conversion_data = calculate_conversion_points(rounds, founders_shares)
    for name, data in conversion_data.items():
        cp = data['conversion_point']
        fig.add_vline(x=cp, line_dash="dash", line_color=colors.get(name, '#64748b'),
                      annotation_text=f"{name} CP", annotation_position="top")
    
    fig.update_layout(
        title=dict(text='Exit Diagram (Composite)', font=dict(size=18, color='#f8fafc')),
        xaxis=dict(title='Exit Value', titlefont=dict(color='#94a3b8'),
                   tickfont=dict(color='#64748b'), gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Payoff', titlefont=dict(color='#94a3b8'),
                   tickfont=dict(color='#64748b'), gridcolor='rgba(255,255,255,0.05)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(bgcolor='rgba(20,20,30,0.8)', font=dict(color='#f8fafc')),
        hovermode='x unified'
    )
    
    return fig

def create_individual_exit_diagrams(rounds: List[RoundInput], founders_shares: float,
                                    max_exit: float = None) -> go.Figure:
    """
    개별 Series Exit Diagram
    📌 강의자료: SERIES DIAGRAMS (p.25)
    """
    active_rounds = [r for r in rounds if r.active]
    n_plots = len(active_rounds) + 1  # +1 for founders
    
    if max_exit is None:
        conversion_data = calculate_conversion_points(rounds, founders_shares)
        max_cp = max([d['conversion_point'] for d in conversion_data.values()], default=100)
        max_exit = max_cp * 1.5
    
    exit_values = np.linspace(0, max_exit, 200)
    
    fig = make_subplots(rows=1, cols=min(n_plots, 4), 
                        subplot_titles=['창업자'] + [r.name for r in active_rounds[:3]],
                        horizontal_spacing=0.08)
    
    colors = {
        'founders': '#10b981',
        'Series A': '#6366f1',
        'Series B': '#f97316',
        'Series C': '#22c55e',
    }
    
    all_parties = ['founders'] + [r.name for r in active_rounds]
    
    for idx, party in enumerate(all_parties[:4]):
        payoffs = []
        for ev in exit_values:
            p = calculate_exit_payoffs(ev, rounds, founders_shares)
            if party in p:
                payoffs.append(p[party]['total'])
            else:
                payoffs.append(0)
        
        fig.add_trace(
            go.Scatter(x=exit_values, y=payoffs, 
                      line=dict(width=2, color=colors.get(party, '#64748b')),
                      showlegend=False),
            row=1, col=idx+1
        )
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc')
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    
    return fig

# =============================================================================
# 유틸리티 함수
# =============================================================================
def format_currency(value: float, suffix: str = 'M') -> str:
    if abs(value) >= 1000:
        return f"{value/1000:,.1f}B"
    return f"{value:,.2f}{suffix}"

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
        <h1>🚀 VC Term Sheet Analyzer v2.0</h1>
        <p>Term Sheet 조건 분석 | Exit Diagram | GP/LP 수익 시뮬레이션 | Option Pricing Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("## ⚙️ Global Settings")
        
        st.markdown("### 👤 Founders")
        st.session_state.global_input.founders_shares = st.number_input(
            "Founders' Shares (M)",
            min_value=1.0, max_value=100.0,
            value=10.0, step=1.0
        )
        
        st.markdown("### 💰 Valuation")
        st.session_state.global_input.total_valuation = st.number_input(
            "Total Valuation",
            min_value=10.0, max_value=10000.0,
            value=100.0, step=10.0
        )
        
        st.markdown("### 📊 Option Parameters")
        st.caption("📖 Base-Case Assumptions (Cochrane, 2005)")
        
        st.session_state.global_input.volatility = st.slider(
            "Volatility (%)", 20, 150, 90,
            help="스타트업 변동성: 보통 90% (Cochrane, 2005)"
        )
        
        st.session_state.global_input.risk_free_rate = st.slider(
            "Risk Free Rate (%)", 0.0, 10.0, 5.0, 0.5
        )
        
        st.session_state.global_input.holding_period = st.slider(
            "Expected Holding Period (yrs)", 1, 10, 5,
            help="Series A: 5년, Series B: 4년, Series C+: 3년"
        )
        
        st.markdown("---")
        st.markdown("### 🏦 Fund Info (for LP Valuation)")
        
        st.session_state.fund_input.committed_capital = st.number_input(
            "Committed Capital", value=100.0, step=10.0
        )
        
        st.session_state.fund_input.lifetime_fees = st.number_input(
            "Lifetime Fees", value=20.0, step=5.0
        )
        
        st.session_state.fund_input.gp_percent = st.slider(
            "GP% (Carry)", 0, 30, 20
        )
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Term Sheet 입력", "📊 Exit Diagram", "💼 Valuation 분석", "📖 사용법"
    ])
    
    # =========================================================================
    # TAB 1: Term Sheet 입력
    # =========================================================================
    with tab1:
        st.markdown("### 📝 EXIT DIAGRAM INPUTS")
        st.caption("vcvtools.com/auto.php 방식의 입력")
        
        # 라운드 활성화
        cols = st.columns(6)
        for idx, r in enumerate(st.session_state.rounds):
            with cols[idx]:
                r.active = st.checkbox(r.name, value=r.active, key=f"active_{r.name}")
        
        st.markdown("---")
        
        # 활성 라운드 입력 테이블
        active_rounds = [r for r in st.session_state.rounds if r.active]
        
        if active_rounds:
            # 테이블 형식 입력
            st.markdown("#### 라운드별 조건 입력")
            
            cols = st.columns(len(active_rounds) + 1)
            
            # 헤더
            with cols[0]:
                st.markdown("**항목**")
                st.markdown("Security Type")
                st.markdown("Investment")
                st.markdown("Shares (M)")
                st.markdown("Liquidation Pref")
            
            for idx, r in enumerate(active_rounds):
                with cols[idx + 1]:
                    badge_class = r.name.lower().replace(" ", "-")
                    st.markdown(f"<span class='series-badge {badge_class}'>{r.name}</span>", 
                               unsafe_allow_html=True)
                    
                    r.security_type = st.selectbox(
                        "Type", ["CP", "RP", "PCP"], 
                        key=f"type_{r.name}",
                        label_visibility="collapsed"
                    )
                    
                    r.investment = st.number_input(
                        "Inv", min_value=0.0, max_value=1000.0, value=float(r.investment),
                        step=1.0, key=f"inv_{r.name}", label_visibility="collapsed"
                    )
                    
                    r.shares = st.number_input(
                        "Shares", min_value=0.0, max_value=100.0, value=float(r.shares),
                        step=1.0, key=f"shares_{r.name}", label_visibility="collapsed"
                    )
                    
                    r.liquidation_pref = st.number_input(
                        "LP", min_value=1.0, max_value=5.0, value=float(r.liquidation_pref),
                        step=0.5, key=f"lp_{r.name}", label_visibility="collapsed"
                    )
            
            st.markdown("---")
            
            # 요약: RVPS 및 Conversion Order
            valid_rounds = [r for r in active_rounds if r.shares > 0]
            
            if valid_rounds:
                st.markdown("### 📋 Conversion Order (RVPS 기준)")
                st.caption("📖 강의자료: Conversion-Order Shortcut (p.28)")
                
                conversion_order = get_conversion_order(st.session_state.rounds)
                
                # RVPS 테이블
                rvps_data = []
                for name, rvps in conversion_order:
                    r = next(r for r in st.session_state.rounds if r.name == name)
                    rvps_data.append({
                        'Series': name,
                        'Investment': r.investment,
                        'Shares (M)': r.shares,
                        'Liq Pref': f"{r.liquidation_pref}x",
                        'RV': r.redemption_value,
                        'RVPS': f"${rvps:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(rvps_data), use_container_width=True, hide_index=True)
                
                # Conversion Order 표시
                order_str = " → ".join([name for name, _ in conversion_order])
                st.markdown(f"""
                <div class="conversion-order">
                    <strong>전환 순서:</strong> {order_str}
                </div>
                """, unsafe_allow_html=True)
                
                st.caption("💡 RVPS가 낮을수록 먼저 전환 (전환이 유리한 시점이 빨리 옴)")
        else:
            st.info("👆 위에서 분석할 Series를 선택하세요.")
    
    # =========================================================================
    # TAB 2: Exit Diagram
    # =========================================================================
    with tab2:
        st.markdown("### 📊 Exit Diagrams")
        st.caption("📖 강의자료: 전환 또는 상환 결정 (p.5), Exit Valuation of CP (p.6)")
        
        valid_rounds = [r for r in st.session_state.rounds if r.active and r.shares > 0]
        
        if not valid_rounds:
            st.info("📝 Term Sheet 입력 탭에서 라운드 정보를 입력하세요.")
        else:
            # 전환점 정보
            conversion_data = calculate_conversion_points(
                st.session_state.rounds, 
                st.session_state.global_input.founders_shares
            )
            
            # Conversion Points 표시
            st.markdown("#### 전환 포인트 (Conversion Points)")
            
            cp_cols = st.columns(len(conversion_data))
            for idx, (name, data) in enumerate(conversion_data.items()):
                with cp_cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{name}</div>
                        <div class="metric-value">{data['conversion_point']:.1f}</div>
                        <div class="metric-sub">RVPS: ${data['rvps']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 개별 Exit Diagram
            st.markdown("#### Series Diagrams")
            fig_individual = create_individual_exit_diagrams(
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            st.plotly_chart(fig_individual, use_container_width=True)
            
            # Composite Exit Diagram
            st.markdown("#### Composite Diagram")
            fig_composite = create_exit_diagram(
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            st.plotly_chart(fig_composite, use_container_width=True)
            
            # 특정 Exit Value에서의 분배
            st.markdown("---")
            st.markdown("#### 특정 Exit Value에서의 분배")
            
            max_cp = max([d['conversion_point'] for d in conversion_data.values()])
            exit_val = st.slider(
                "Exit Value",
                min_value=0.0,
                max_value=float(max_cp * 2),
                value=float(st.session_state.global_input.total_valuation)
            )
            
            payoffs = calculate_exit_payoffs(
                exit_val,
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            
            payoff_cols = st.columns(len(payoffs))
            for idx, (party, data) in enumerate(payoffs.items()):
                with payoff_cols[idx]:
                    display_name = '창업자' if party == 'founders' else party
                    pct = (data['total'] / exit_val * 100) if exit_val > 0 else 0
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{display_name}</div>
                        <div class="metric-value">{data['total']:.2f}</div>
                        <div class="metric-sub">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 3: Valuation 분석
    # =========================================================================
    with tab3:
        st.markdown("### 💼 AUTO OUTPUTS")
        st.caption("📖 강의자료: Option Pricing Model로 가치 산정 (p.8)")
        
        valid_rounds = [r for r in st.session_state.rounds if r.active and r.shares > 0]
        
        if not valid_rounds:
            st.info("📝 Term Sheet 입력 탭에서 라운드 정보를 입력하세요.")
        else:
            # Partial Valuation 공식
            st.markdown("#### Partial Valuation 공식")
            st.caption("📖 강의자료: Value of CP = V - C(K1) + α×C(K2) (p.11)")
            
            formula_data = calculate_partial_valuation_formula(
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            
            for name, data in formula_data.items():
                st.markdown(f"**{name}** (전환순서 #{data['order']})")
                st.markdown(f"""
                <div class="formula-box">
                Partial Valuation = {data['formula']}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # AUTO OUTPUTS 테이블
            st.markdown("#### Valuation Results")
            
            results_data = []
            for r in valid_rounds:
                partial_val = calculate_partial_valuation(
                    r, st.session_state.rounds,
                    st.session_state.global_input.founders_shares,
                    st.session_state.global_input,
                    use_re_option=True
                )
                
                gp_lp = calculate_gp_lp_valuation(
                    partial_val,
                    st.session_state.fund_input,
                    r.investment
                )
                
                results_data.append({
                    'Series': r.name,
                    'LP Cost': f"{gp_lp['lp_cost']:.2f}",
                    'Partial Valuation': f"{gp_lp['partial_valuation']:.4f}",
                    'GP Valuation': f"{gp_lp['gp_valuation']:.4f}",
                    'LP Valuation': f"{gp_lp['lp_valuation']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
            
            # Implied-post Valuation (Calculate Breakeven)
            st.markdown("---")
            st.markdown("#### Implied-post Valuation")
            
            if st.button("Calculate Breakeven", type="primary"):
                # 간단한 근사: LP Cost = LP Valuation이 되는 Total Valuation 찾기
                target_round = valid_rounds[-1]  # 최신 라운드
                lp_cost = calculate_lp_cost(st.session_state.fund_input, target_round.investment)
                
                # Binary search for breakeven
                low, high = 10, 1000
                for _ in range(50):
                    mid = (low + high) / 2
                    test_global = GlobalInput(
                        founders_shares=st.session_state.global_input.founders_shares,
                        total_valuation=mid,
                        volatility=st.session_state.global_input.volatility,
                        risk_free_rate=st.session_state.global_input.risk_free_rate,
                        holding_period=st.session_state.global_input.holding_period
                    )
                    partial_val = calculate_partial_valuation(
                        target_round, st.session_state.rounds,
                        test_global.founders_shares, test_global
                    )
                    lp_val = partial_val * (1 - st.session_state.fund_input.gp_percent / 100)
                    
                    if lp_val < lp_cost:
                        low = mid
                    else:
                        high = mid
                
                st.success(f"**Implied-post Valuation:** {mid:.4f}")
    
    # =========================================================================
    # TAB 4: 사용법
    # =========================================================================
    with tab4:
        st.markdown("### 📖 사용 가이드")
        
        st.markdown("""
        #### 🎯 도구 개요
        
        이 도구는 Yale 경영대학원 Andrew Metrick 교수와 UC Davis Ayako Yasuda 교수의 
        **"Venture Capital and the Finance of Innovation"** 교재의 VCV Tools를 
        한국 시장에 맞게 구현한 것입니다.
        
        ---
        
        #### 📊 주요 개념
        
        **1. RVPS (Redemption Value Per Share)**
        ```
        RVPS = 상환가치(RV) / 전환 시 받을 주식수
        ```
        - RVPS가 낮을수록 먼저 전환 (전환이 유리한 시점이 빨리 옴)
        - Conversion Order 결정의 핵심
        
        **2. Conversion Point (전환점)**
        ```
        전환 조건: 지분율 × (기업가치 - 선순위 RV) > 나의 RV
        ```
        - 이 조건을 만족하는 최소 기업가치가 전환점
        
        **3. Partial Valuation**
        ```
        CP 가치 = V - C(K₁) + α×C(K₂)
        ```
        - V: 기업가치
        - C(K): Strike K인 콜옵션 가치
        - Random Expiration (RE) Option으로 계산
        
        **4. LP/GP Valuation**
        ```
        LP Cost = (Committed Capital / Investable) × Investment
        GP Valuation = Partial Valuation × GP%
        LP Valuation = Partial Valuation - GP Valuation
        ```
        
        ---
        
        #### 🔧 Option Pricing Assumptions (기본값)
        
        | 파라미터 | 값 | 출처 |
        |---------|-----|------|
        | Volatility | 90% | Cochrane (2005) |
        | Risk-free Rate | 5% | 교재 기본값 |
        | Holding Period | Series A: 5년, B: 4년, C+: 3년 | 교재 기본값 |
        
        ---
        
        #### 📚 참고 자료
        
        - **원본**: [vcvtools.com](http://vcvtools.com/)
        - **교재**: Metrick & Yasuda, *Venture Capital and the Finance of Innovation* (2nd Ed.)
        - **강의**: Ch9 & 14 Preferred Stock, Ch15 Late Round Investment
        """)
        
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #6366f1;">🏢 인프라프론티어자산운용(주)</h4>
            <p style="color: #94a3b8;">VC Term Sheet Analyzer v2.0</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
