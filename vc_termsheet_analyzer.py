"""
VC Term Sheet Analyzer v1.0 - 한국 VC 시장용
벤치마킹: vcvtools.com/auto.php
인프라프론티어자산운용(주)

핵심 기능:
1. Term Sheet 입력 → 지분 분배 계산
2. Series A~H 라운드별 분석
3. GP/LP 수익 분배 시뮬레이션
4. Exit Diagram (Payoff Schedule) 시각화
5. 한국 VC 시장 맞춤 (원화, 한국 용어)
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
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict, Optional
import math

# =============================================================================
# CSS 스타일 - 현대적 다크 테마 + 글래스모피즘
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
        --accent-info: #06b6d4;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        --gradient-dark: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
    }
    
    .stApp {
        background: var(--gradient-dark);
        font-family: 'Outfit', sans-serif;
    }
    
    /* 메인 헤더 */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid var(--border-accent);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
    }
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* 글래스 카드 */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.1);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(145deg, rgba(20, 20, 30, 0.9) 0%, rgba(15, 15, 25, 0.9) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover::after {
        opacity: 1;
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
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    .metric-change {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-change.positive { color: var(--accent-success); }
    .metric-change.negative { color: var(--accent-danger); }
    .metric-change.neutral { color: var(--text-muted); }
    
    /* 라운드 토글 버튼 */
    .round-toggle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 2px solid transparent;
    }
    .round-toggle.inactive {
        background: var(--bg-glass);
        color: var(--text-muted);
        border-color: var(--border-subtle);
    }
    .round-toggle.active {
        background: var(--gradient-primary);
        color: white;
        border-color: var(--accent-primary);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* 섹션 헤더 */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .section-header h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: var(--text-primary);
        margin: 0;
    }
    .section-header .icon {
        font-size: 1.25rem;
    }
    
    /* 입력 그룹 */
    .input-group {
        background: rgba(15, 15, 25, 0.6);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .input-group-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--accent-primary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }
    
    /* 결과 테이블 */
    .result-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.9rem;
    }
    .result-table th {
        background: rgba(99, 102, 241, 0.1);
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.75rem 1rem;
        text-align: left;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border-subtle);
    }
    .result-table td {
        padding: 0.75rem 1rem;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-subtle);
        font-family: 'JetBrains Mono', monospace;
    }
    .result-table tr:hover td {
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* 차트 컨테이너 */
    .chart-container {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .chart-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    /* 툴팁 스타일 */
    .tooltip-text {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    /* 배지 */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-primary {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
    }
    
    /* 탭 커스텀 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-glass);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        border-color: var(--accent-primary) !important;
    }
    
    /* 사이드바 */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: var(--bg-secondary);
    }
    
    /* Selectbox, Input 스타일 */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(15, 15, 25, 0.8) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    /* 버튼 */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* 숨김 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* 시리즈 컬러 */
    .series-a { --series-color: #6366f1; }
    .series-b { --series-color: #8b5cf6; }
    .series-c { --series-color: #a855f7; }
    .series-d { --series-color: #d946ef; }
    .series-e { --series-color: #ec4899; }
    .series-f { --series-color: #f43f5e; }
    
    /* 스크롤바 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-subtle);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 데이터 클래스 정의
# =============================================================================
@dataclass
class RoundInput:
    """투자 라운드 입력 데이터"""
    name: str  # Series A, B, C...
    active: bool = False
    security_type: str = "CP"  # CP, RP, PCP, PCPC
    investment: float = 0  # 투자금액 (억원)
    shares: float = 0  # 주식 수
    liquidation_pref: float = 1.0  # 청산우선권 배수
    participation_cap: float = 0  # 참가권 상한 (0=무제한)
    anti_dilution: str = "None"  # None, Full, Weighted
    
@dataclass
class FundInput:
    """펀드 입력 데이터"""
    committed_capital: float = 0  # 약정총액 (억원)
    management_fee: float = 2.0  # 관리보수 (%)
    carried_interest: float = 20.0  # 성과보수 (%)
    hurdle_rate: float = 8.0  # 허들레이트 (%)

@dataclass
class GlobalInput:
    """글로벌 입력 데이터"""
    founders_shares: float = 10000000  # 창업자 주식 수
    total_valuation: float = 100  # 총 기업가치 (억원)
    volatility: float = 80  # 변동성 (%)
    risk_free_rate: float = 3.5  # 무위험이자율 (%)
    holding_period: float = 5  # 예상 보유기간 (년)
    exit_value: float = 500  # Exit 가치 (억원)

# =============================================================================
# 핵심 계산 함수들
# =============================================================================
def calculate_ownership(rounds: List[RoundInput], founders_shares: float) -> Dict:
    """지분율 계산"""
    total_shares = founders_shares
    results = {'founders': {'shares': founders_shares, 'ownership': 0}}
    
    for r in rounds:
        if r.active and r.shares > 0:
            total_shares += r.shares
            results[r.name] = {'shares': r.shares, 'ownership': 0, 'investment': r.investment}
    
    # 지분율 계산
    if total_shares > 0:
        results['founders']['ownership'] = founders_shares / total_shares * 100
        for r in rounds:
            if r.active and r.name in results:
                results[r.name]['ownership'] = r.shares / total_shares * 100
    
    results['total_shares'] = total_shares
    return results

def calculate_post_money_valuation(investment: float, ownership_pct: float) -> float:
    """Post-money 밸류에이션 계산"""
    if ownership_pct > 0:
        return investment / (ownership_pct / 100)
    return 0

def calculate_payoff_schedule(exit_value: float, rounds: List[RoundInput], 
                              founders_shares: float) -> pd.DataFrame:
    """
    Exit 가치에 따른 Payoff Schedule 계산
    각 이해관계자별 수령액 계산
    """
    ownership = calculate_ownership(rounds, founders_shares)
    total_shares = ownership['total_shares']
    
    # 정렬: 최신 라운드부터 청산우선권 행사
    active_rounds = [r for r in rounds if r.active]
    active_rounds.reverse()  # 최신 라운드 우선
    
    remaining_value = exit_value
    payoffs = {}
    
    # Step 1: 청산우선권 분배 (Liquidation Preference)
    for r in active_rounds:
        liq_pref = r.investment * r.liquidation_pref
        
        if r.security_type in ["RP", "CP", "PCP", "PCPC"]:
            # 청산우선권 우선 지급
            pref_payout = min(liq_pref, remaining_value)
            payoffs[r.name] = {'preference': pref_payout, 'participation': 0, 'conversion': 0}
            remaining_value -= pref_payout
    
    # Step 2: 참가권 분배 (Participation) 또는 전환권 (Conversion)
    for r in active_rounds:
        if r.name not in payoffs:
            payoffs[r.name] = {'preference': 0, 'participation': 0, 'conversion': 0}
        
        if r.security_type in ["PCP", "PCPC"]:
            # 참가형: 잔여가치에서 지분율만큼 추가 수령
            if remaining_value > 0:
                ownership_pct = ownership.get(r.name, {}).get('ownership', 0) / 100
                participation = remaining_value * ownership_pct
                
                # CAP 적용
                if r.participation_cap > 0:
                    max_return = r.investment * r.participation_cap
                    total_so_far = payoffs[r.name]['preference']
                    participation = min(participation, max(0, max_return - total_so_far))
                
                payoffs[r.name]['participation'] = participation
    
    # Step 3: 전환 vs 청산우선권 비교 (CP의 경우)
    for r in active_rounds:
        if r.security_type == "CP":
            # 전환 시 수령액
            ownership_pct = ownership.get(r.name, {}).get('ownership', 0) / 100
            conversion_value = exit_value * ownership_pct
            
            # 청산우선권 vs 전환 중 큰 값 선택
            pref_value = payoffs[r.name]['preference']
            if conversion_value > pref_value:
                payoffs[r.name] = {'preference': 0, 'participation': 0, 'conversion': conversion_value}
    
    # Step 4: 창업자 몫 계산
    total_investor_payout = sum(
        p['preference'] + p['participation'] + p['conversion'] 
        for p in payoffs.values()
    )
    founders_payout = max(0, exit_value - total_investor_payout)
    payoffs['founders'] = {'preference': 0, 'participation': 0, 'conversion': founders_payout}
    
    return payoffs

def calculate_gp_lp_split(fund: FundInput, round_value: float, round_investment: float) -> Dict:
    """GP/LP 수익 분배 계산"""
    # 투자가능금액 (약정총액 - 관리보수)
    investable = fund.committed_capital * (1 - fund.management_fee / 100 * 10)  # 10년 가정
    
    # LP 투자비용
    lp_cost = round_investment / investable * fund.committed_capital if investable > 0 else 0
    
    # 수익
    profit = round_value - round_investment
    
    if profit <= 0:
        return {
            'lp_cost': lp_cost,
            'profit': profit,
            'hurdle': 0,
            'gp_carry': 0,
            'lp_return': round_value,
            'gp_total': 0,
            'lp_total': round_value
        }
    
    # 허들레이트 초과분에 대해 Carry 계산
    hurdle_amount = lp_cost * (fund.hurdle_rate / 100) * 5  # 5년 가정
    excess_profit = max(0, profit - hurdle_amount)
    
    gp_carry = excess_profit * (fund.carried_interest / 100)
    lp_return = round_value - gp_carry
    
    return {
        'lp_cost': lp_cost,
        'profit': profit,
        'hurdle': hurdle_amount,
        'gp_carry': gp_carry,
        'lp_return': lp_return,
        'gp_total': gp_carry,
        'lp_total': lp_return
    }

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes 콜옵션 가치 계산"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_value = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_value

def calculate_partial_valuation(rounds: List[RoundInput], global_input: GlobalInput) -> Dict:
    """
    부분가치평가 (옵션 가격 모델 활용)
    각 라운드의 현재 가치를 옵션으로 평가
    """
    ownership = calculate_ownership(rounds, global_input.founders_shares)
    total_shares = ownership['total_shares']
    
    results = {}
    cumulative_strike = 0
    
    for r in rounds:
        if r.active:
            # 해당 라운드의 Strike Price (누적 청산우선권)
            cumulative_strike += r.investment * r.liquidation_pref
            
            # 해당 라운드의 지분 비율
            ownership_pct = ownership.get(r.name, {}).get('ownership', 0) / 100
            
            # 옵션으로서의 가치 계산
            if r.security_type == "CP":
                # 전환우선주: 콜옵션
                option_value = black_scholes_call(
                    S=global_input.total_valuation * ownership_pct,
                    K=cumulative_strike * ownership_pct,
                    T=global_input.holding_period,
                    r=global_input.risk_free_rate / 100,
                    sigma=global_input.volatility / 100
                )
                results[r.name] = {
                    'option_value': option_value,
                    'ownership_pct': ownership_pct * 100,
                    'strike': cumulative_strike * ownership_pct
                }
            else:
                # 상환우선주 등: 단순 지분가치
                results[r.name] = {
                    'option_value': global_input.total_valuation * ownership_pct,
                    'ownership_pct': ownership_pct * 100,
                    'strike': 0
                }
    
    return results

# =============================================================================
# 시각화 함수들
# =============================================================================
def create_exit_diagram(exit_values: np.array, rounds: List[RoundInput], 
                        founders_shares: float) -> go.Figure:
    """Exit Diagram (Payoff Chart) 생성"""
    
    # 각 Exit 가치별 Payoff 계산
    series_names = ['founders'] + [r.name for r in rounds if r.active]
    payoff_data = {name: [] for name in series_names}
    
    for ev in exit_values:
        payoffs = calculate_payoff_schedule(ev, rounds, founders_shares)
        for name in series_names:
            if name in payoffs:
                total = payoffs[name]['preference'] + payoffs[name]['participation'] + payoffs[name]['conversion']
                payoff_data[name].append(total)
            else:
                payoff_data[name].append(0)
    
    # Plotly Figure 생성
    fig = go.Figure()
    
    colors = {
        'founders': '#10b981',
        'Series A': '#6366f1',
        'Series B': '#8b5cf6',
        'Series C': '#a855f7',
        'Series D': '#d946ef',
        'Series E': '#ec4899',
        'Series F': '#f43f5e',
    }
    
    for name in series_names:
        color = colors.get(name, '#64748b')
        fig.add_trace(go.Scatter(
            x=exit_values,
            y=payoff_data[name],
            name=name if name != 'founders' else '창업자',
            mode='lines',
            line=dict(width=3, color=color),
            fill='tonexty' if name != 'founders' else None,
            hovertemplate=f'<b>{name}</b><br>Exit: %{{x:.0f}}억<br>Payoff: %{{y:.1f}}억<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Exit Value별 Payoff Schedule',
            font=dict(size=18, color='#f8fafc', family='Outfit')
        ),
        xaxis=dict(
            title='Exit Value (억원)',
            titlefont=dict(color='#94a3b8'),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Payoff (억원)',
            titlefont=dict(color='#94a3b8'),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            bgcolor='rgba(20,20,30,0.8)',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(color='#f8fafc')
        ),
        hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    return fig

def create_ownership_pie(ownership: Dict) -> go.Figure:
    """지분 구조 파이 차트"""
    labels = []
    values = []
    colors = []
    
    color_map = {
        'founders': '#10b981',
        'Series A': '#6366f1',
        'Series B': '#8b5cf6',
        'Series C': '#a855f7',
        'Series D': '#d946ef',
        'Series E': '#ec4899',
        'Series F': '#f43f5e',
    }
    
    for key, data in ownership.items():
        if key != 'total_shares' and isinstance(data, dict) and data.get('ownership', 0) > 0:
            label = '창업자' if key == 'founders' else key
            labels.append(label)
            values.append(data['ownership'])
            colors.append(color_map.get(key, '#64748b'))
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#0a0a0f', width=2)),
        textinfo='label+percent',
        textfont=dict(color='#f8fafc', size=12),
        hovertemplate='<b>%{label}</b><br>지분율: %{percent}<br>%{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text='지분 구조',
            font=dict(size=16, color='#f8fafc', family='Outfit')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            bgcolor='rgba(20,20,30,0.8)',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(color='#f8fafc')
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[dict(
            text='지분율',
            x=0.5, y=0.5,
            font=dict(size=14, color='#64748b'),
            showarrow=False
        )]
    )
    
    return fig

def create_gp_lp_waterfall(gp_lp: Dict, round_name: str) -> go.Figure:
    """GP/LP 수익 분배 워터폴 차트"""
    
    fig = go.Figure(go.Waterfall(
        name="수익 분배",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["투자금액", "수익", "GP Carry", "LP 수익", "총 분배"],
        textposition="outside",
        text=[
            f"{gp_lp['lp_cost']:.1f}억",
            f"+{gp_lp['profit']:.1f}억" if gp_lp['profit'] >= 0 else f"{gp_lp['profit']:.1f}억",
            f"-{gp_lp['gp_carry']:.1f}억",
            f"{gp_lp['lp_return']:.1f}억",
            f"{gp_lp['lp_total']:.1f}억"
        ],
        y=[gp_lp['lp_cost'], gp_lp['profit'], -gp_lp['gp_carry'], 0, gp_lp['lp_total']],
        connector={"line": {"color": "rgba(99, 102, 241, 0.5)"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#6366f1"}}
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{round_name} GP/LP 수익 분배',
            font=dict(size=16, color='#f8fafc', family='Outfit')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickfont=dict(color='#94a3b8'),
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            title='금액 (억원)',
            titlefont=dict(color='#94a3b8'),
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(255,255,255,0.05)'
        ),
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False
    )
    
    return fig

# =============================================================================
# 유틸리티 함수
# =============================================================================
def format_currency(value: float, unit: str = '억원') -> str:
    """통화 포맷팅"""
    if abs(value) >= 10000:
        return f"{value/10000:,.1f}조원"
    return f"{value:,.1f}{unit}"

def format_percent(value: float) -> str:
    """퍼센트 포맷팅"""
    return f"{value:.2f}%"

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
        <h1>🚀 VC Term Sheet Analyzer</h1>
        <p>Term Sheet 조건 분석 및 LP/GP 수익 시뮬레이션 도구 | 한국 VC 시장 맞춤</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("## ⚙️ 글로벌 설정")
        
        st.markdown("### 👤 창업자 정보")
        st.session_state.global_input.founders_shares = st.number_input(
            "창업자 보통주 (주)",
            min_value=1000,
            max_value=100000000,
            value=10000000,
            step=100000,
            format="%d"
        )
        
        st.markdown("### 💰 기업가치")
        st.session_state.global_input.total_valuation = st.number_input(
            "현재 기업가치 (억원)",
            min_value=1.0,
            max_value=100000.0,
            value=100.0,
            step=10.0
        )
        
        st.session_state.global_input.exit_value = st.number_input(
            "예상 Exit 가치 (억원)",
            min_value=1.0,
            max_value=100000.0,
            value=500.0,
            step=50.0
        )
        
        st.markdown("### 📊 옵션 파라미터")
        st.session_state.global_input.volatility = st.slider(
            "변동성 (%)",
            min_value=20,
            max_value=150,
            value=80
        )
        
        st.session_state.global_input.risk_free_rate = st.slider(
            "무위험이자율 (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1
        )
        
        st.session_state.global_input.holding_period = st.slider(
            "예상 보유기간 (년)",
            min_value=1,
            max_value=15,
            value=5
        )
        
        st.markdown("---")
        
        st.markdown("### 🏦 펀드 정보")
        st.session_state.fund_input.committed_capital = st.number_input(
            "약정총액 (억원)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=50.0
        )
        
        st.session_state.fund_input.management_fee = st.slider(
            "관리보수 (%)",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
        
        st.session_state.fund_input.carried_interest = st.slider(
            "성과보수 (%)",
            min_value=0.0,
            max_value=30.0,
            value=20.0,
            step=1.0
        )
        
        st.session_state.fund_input.hurdle_rate = st.slider(
            "허들레이트 (%)",
            min_value=0.0,
            max_value=15.0,
            value=8.0,
            step=0.5
        )
    
    # 메인 콘텐츠
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Term Sheet 입력", "📊 Exit Diagram", "💼 GP/LP 분석", "📖 사용법"
    ])
    
    # =========================================================================
    # TAB 1: Term Sheet 입력
    # =========================================================================
    with tab1:
        st.markdown("""
        <div class="section-header">
            <span class="icon">📝</span>
            <h3>투자 라운드 정보 입력</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("투자 라운드를 선택하고 Term Sheet 정보를 입력하세요.")
        
        # 라운드 선택 토글
        st.markdown("#### 활성화할 라운드 선택")
        cols = st.columns(6)
        for idx, r in enumerate(st.session_state.rounds):
            with cols[idx]:
                r.active = st.checkbox(r.name, value=r.active, key=f"toggle_{r.name}")
        
        st.markdown("---")
        
        # 활성화된 라운드별 입력
        active_rounds = [r for r in st.session_state.rounds if r.active]
        
        if not active_rounds:
            st.info("👆 위에서 분석할 투자 라운드를 선택하세요.")
        else:
            for r in active_rounds:
                with st.expander(f"🔵 {r.name} 상세 입력", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        r.security_type = st.selectbox(
                            "증권 유형",
                            ["CP", "RP", "PCP", "PCPC"],
                            index=["CP", "RP", "PCP", "PCPC"].index(r.security_type),
                            key=f"security_{r.name}",
                            help="CP: 전환우선주, RP: 상환우선주, PCP: 참가적 전환우선주, PCPC: 참가적 전환우선주(Cap)"
                        )
                        
                        r.investment = st.number_input(
                            "투자금액 (억원)",
                            min_value=0.0,
                            max_value=10000.0,
                            value=float(r.investment),
                            step=1.0,
                            key=f"investment_{r.name}"
                        )
                    
                    with col2:
                        r.shares = st.number_input(
                            "발행주식수 (주)",
                            min_value=0,
                            max_value=100000000,
                            value=int(r.shares),
                            step=10000,
                            key=f"shares_{r.name}"
                        )
                        
                        r.liquidation_pref = st.number_input(
                            "청산우선권 배수",
                            min_value=0.0,
                            max_value=5.0,
                            value=float(r.liquidation_pref),
                            step=0.1,
                            key=f"liq_pref_{r.name}",
                            help="1.0 = 1x, 2.0 = 2x"
                        )
                    
                    with col3:
                        r.participation_cap = st.number_input(
                            "참가권 상한 (배수, 0=무제한)",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(r.participation_cap),
                            step=0.5,
                            key=f"cap_{r.name}"
                        )
                        
                        r.anti_dilution = st.selectbox(
                            "희석방지조항",
                            ["None", "Full Ratchet", "Weighted Average"],
                            index=["None", "Full Ratchet", "Weighted Average"].index(r.anti_dilution),
                            key=f"anti_dilution_{r.name}"
                        )
            
            st.markdown("---")
            
            # 요약 결과
            st.markdown("""
            <div class="section-header">
                <span class="icon">📊</span>
                <h3>분석 결과 요약</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 지분 구조 계산
            ownership = calculate_ownership(st.session_state.rounds, 
                                           st.session_state.global_input.founders_shares)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # 지분 구조 파이 차트
                fig_pie = create_ownership_pie(ownership)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # 요약 메트릭
                st.markdown("#### 💰 밸류에이션")
                
                total_investment = sum(r.investment for r in active_rounds)
                total_investor_ownership = sum(
                    ownership.get(r.name, {}).get('ownership', 0) 
                    for r in active_rounds
                )
                
                if total_investor_ownership > 0:
                    implied_post = total_investment / (total_investor_ownership / 100)
                else:
                    implied_post = 0
                
                implied_pre = implied_post - total_investment
                
                mcol1, mcol2, mcol3 = st.columns(3)
                
                with mcol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">총 투자금액</div>
                        <div class="metric-value">{format_currency(total_investment)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with mcol2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Post-Money</div>
                        <div class="metric-value">{format_currency(implied_post)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with mcol3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Pre-Money</div>
                        <div class="metric-value">{format_currency(implied_pre)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### 📋 지분 내역")
                
                # 지분 테이블
                table_data = []
                table_data.append({
                    '구분': '창업자',
                    '주식수': f"{ownership['founders']['shares']:,.0f}",
                    '지분율': f"{ownership['founders']['ownership']:.2f}%",
                    '투자금액': '-'
                })
                
                for r in active_rounds:
                    if r.name in ownership:
                        table_data.append({
                            '구분': r.name,
                            '주식수': f"{ownership[r.name]['shares']:,.0f}",
                            '지분율': f"{ownership[r.name]['ownership']:.2f}%",
                            '투자금액': format_currency(r.investment)
                        })
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
    # =========================================================================
    # TAB 2: Exit Diagram
    # =========================================================================
    with tab2:
        st.markdown("""
        <div class="section-header">
            <span class="icon">📊</span>
            <h3>Exit Diagram (Payoff Schedule)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        Exit 가치에 따라 각 이해관계자(창업자, 투자자)가 받게 되는 금액을 시각화합니다.
        청산우선권, 참가권, 전환권 등 Term Sheet 조건에 따라 분배가 달라집니다.
        """)
        
        active_rounds = [r for r in st.session_state.rounds if r.active]
        
        if not active_rounds:
            st.info("📝 Term Sheet 입력 탭에서 투자 라운드를 활성화하세요.")
        else:
            # Exit 가치 범위 설정
            col1, col2 = st.columns(2)
            with col1:
                exit_min = st.number_input(
                    "최소 Exit 가치 (억원)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=0.0,
                    step=10.0
                )
            with col2:
                exit_max = st.number_input(
                    "최대 Exit 가치 (억원)",
                    min_value=10.0,
                    max_value=50000.0,
                    value=float(st.session_state.global_input.exit_value * 2),
                    step=100.0
                )
            
            # Exit Diagram 생성
            exit_values = np.linspace(exit_min, exit_max, 200)
            fig_exit = create_exit_diagram(
                exit_values, 
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            st.plotly_chart(fig_exit, use_container_width=True, config={'displayModeBar': False})
            
            # 특정 Exit 가치에서의 분배
            st.markdown("---")
            st.markdown("#### 🎯 특정 Exit 가치에서의 분배")
            
            specific_exit = st.slider(
                "Exit 가치 선택 (억원)",
                min_value=float(exit_min),
                max_value=float(exit_max),
                value=float(st.session_state.global_input.exit_value),
                step=10.0
            )
            
            payoffs = calculate_payoff_schedule(
                specific_exit,
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            
            # 분배 결과 표시
            cols = st.columns(len(payoffs))
            
            color_map = {
                'founders': '#10b981',
                'Series A': '#6366f1',
                'Series B': '#8b5cf6',
                'Series C': '#a855f7',
                'Series D': '#d946ef',
                'Series E': '#ec4899',
                'Series F': '#f43f5e',
            }
            
            for idx, (name, data) in enumerate(payoffs.items()):
                with cols[idx]:
                    total_payoff = data['preference'] + data['participation'] + data['conversion']
                    display_name = '창업자' if name == 'founders' else name
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {color_map.get(name, '#64748b')};">
                        <div class="metric-label">{display_name}</div>
                        <div class="metric-value">{format_currency(total_payoff)}</div>
                        <div class="metric-change neutral">
                            {total_payoff / specific_exit * 100:.1f}% of Exit
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 상세 분배 테이블
            st.markdown("#### 📋 상세 분배 내역")
            
            detail_data = []
            for name, data in payoffs.items():
                display_name = '창업자' if name == 'founders' else name
                total = data['preference'] + data['participation'] + data['conversion']
                detail_data.append({
                    '구분': display_name,
                    '청산우선권': format_currency(data['preference']) if data['preference'] > 0 else '-',
                    '참가권': format_currency(data['participation']) if data['participation'] > 0 else '-',
                    '전환권': format_currency(data['conversion']) if data['conversion'] > 0 else '-',
                    '총 수령액': format_currency(total),
                    '비율': f"{total / specific_exit * 100:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
    
    # =========================================================================
    # TAB 3: GP/LP 분석
    # =========================================================================
    with tab3:
        st.markdown("""
        <div class="section-header">
            <span class="icon">💼</span>
            <h3>GP/LP 수익 분배 분석</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        VC 펀드의 GP(업무집행사원)와 LP(유한책임사원) 간의 수익 분배를 분석합니다.
        약정총액, 관리보수, 성과보수, 허들레이트 조건에 따른 분배를 시뮬레이션합니다.
        """)
        
        active_rounds = [r for r in st.session_state.rounds if r.active]
        
        if not active_rounds:
            st.info("📝 Term Sheet 입력 탭에서 투자 라운드를 활성화하세요.")
        elif st.session_state.fund_input.committed_capital == 0:
            st.info("👈 사이드바에서 펀드 정보(약정총액)를 입력하세요.")
        else:
            # 펀드 정보 요약
            st.markdown("#### 🏦 펀드 정보")
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            
            with fcol1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">약정총액</div>
                    <div class="metric-value">{format_currency(st.session_state.fund_input.committed_capital)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with fcol2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">관리보수</div>
                    <div class="metric-value">{st.session_state.fund_input.management_fee}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with fcol3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">성과보수 (Carry)</div>
                    <div class="metric-value">{st.session_state.fund_input.carried_interest}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with fcol4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">허들레이트</div>
                    <div class="metric-value">{st.session_state.fund_input.hurdle_rate}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 라운드별 GP/LP 분석
            st.markdown("#### 📊 라운드별 GP/LP 수익 분배")
            
            # Exit 가치에서의 각 라운드 수령액 계산
            exit_val = st.session_state.global_input.exit_value
            payoffs = calculate_payoff_schedule(
                exit_val,
                st.session_state.rounds,
                st.session_state.global_input.founders_shares
            )
            
            for r in active_rounds:
                if r.name in payoffs:
                    round_payoff = payoffs[r.name]['preference'] + payoffs[r.name]['participation'] + payoffs[r.name]['conversion']
                    
                    st.markdown(f"##### {r.name}")
                    
                    gp_lp = calculate_gp_lp_split(
                        st.session_state.fund_input,
                        round_payoff,
                        r.investment
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # 워터폴 차트
                        fig_waterfall = create_gp_lp_waterfall(gp_lp, r.name)
                        st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})
                    
                    with col2:
                        # 요약 메트릭
                        mcol1, mcol2 = st.columns(2)
                        
                        with mcol1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">투자금액</div>
                                <div class="metric-value">{format_currency(r.investment)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Exit 수령액</div>
                                <div class="metric-value">{format_currency(round_payoff)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with mcol2:
                            multiple = round_payoff / r.investment if r.investment > 0 else 0
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">투자 배수</div>
                                <div class="metric-value">{multiple:.2f}x</div>
                                <div class="metric-change {'positive' if multiple > 1 else 'negative'}">
                                    {'+' if multiple > 1 else ''}{(multiple - 1) * 100:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">GP Carry</div>
                                <div class="metric-value">{format_currency(gp_lp['gp_carry'])}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # LP 수익률
                        lp_return_pct = (gp_lp['lp_return'] - gp_lp['lp_cost']) / gp_lp['lp_cost'] * 100 if gp_lp['lp_cost'] > 0 else 0
                        
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%);">
                            <div class="metric-label">LP 순수익률</div>
                            <div class="metric-value" style="color: {'#10b981' if lp_return_pct > 0 else '#ef4444'};">
                                {lp_return_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
    
    # =========================================================================
    # TAB 4: 사용법
    # =========================================================================
    with tab4:
        st.markdown("""
        <div class="section-header">
            <span class="icon">📖</span>
            <h3>사용 가이드</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 도구 개요
        
        이 도구는 VC 투자의 Term Sheet 조건이 각 이해관계자의 수익에 어떤 영향을 미치는지 
        분석하고 시각화합니다.
        
        ---
        
        ### 📝 주요 기능
        
        #### 1. Term Sheet 입력
        - **증권 유형**: CP(전환우선주), RP(상환우선주), PCP(참가적 전환우선주)
        - **청산우선권**: 청산 시 투자금의 몇 배를 먼저 받는지 (1x, 2x 등)
        - **참가권**: 청산우선권 수령 후 잔여가치에도 참여 가능 여부
        - **희석방지조항**: Full Ratchet, Weighted Average
        
        #### 2. Exit Diagram
        - Exit 가치에 따른 각 이해관계자별 수령액 시각화
        - 청산우선권, 참가권, 전환권 행사 시점 파악
        - 특정 Exit 가치에서의 상세 분배 분석
        
        #### 3. GP/LP 분석
        - VC 펀드의 GP/LP 수익 분배 시뮬레이션
        - 관리보수, 성과보수(Carry), 허들레이트 반영
        - LP 순수익률 계산
        
        ---
        
        ### 📊 용어 설명
        
        | 용어 | 설명 |
        |------|------|
        | **CP (Convertible Preferred)** | 전환우선주. 보통주로 전환 가능한 우선주 |
        | **RP (Redeemable Preferred)** | 상환우선주. 일정 기간 후 상환 청구 가능 |
        | **PCP (Participating CP)** | 참가적 전환우선주. 청산우선권 + 잔여가치 참여 |
        | **청산우선권 (Liquidation Preference)** | 청산/매각 시 우선 수령권 |
        | **Carry (Carried Interest)** | GP의 성과보수 (보통 20%) |
        | **Hurdle Rate** | Carry 지급 전 LP에게 보장하는 최소 수익률 |
        
        ---
        
        ### 💡 활용 팁
        
        1. **스타트업 창업자**: Term Sheet 협상 전 다양한 시나리오 시뮬레이션
        2. **VC 심사역**: 투자 조건별 예상 수익 분석
        3. **LP 담당자**: 펀드 투자 시 예상 수익률 검토
        4. **교육용**: VC 투자 구조 이해
        
        ---
        
        ### 📚 참고 자료
        
        이 도구는 *"Venture Capital and the Finance of Innovation"* (Metrick & Yasuda) 
        교재의 VCV Tools를 한국 시장에 맞게 재구현한 것입니다.
        """)
        
        st.markdown("""
        <div class="glass-card" style="margin-top: 2rem;">
            <h4 style="color: #6366f1; margin-bottom: 0.5rem;">🏢 인프라프론티어자산운용(주)</h4>
            <p style="color: #94a3b8; margin: 0;">VC Term Sheet Analyzer v1.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 푸터
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0; margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.05);">
        🚀 VC Term Sheet Analyzer v1.0 | 인프라프론티어자산운용(주) | 
        Powered by vcvtools.com methodology
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
