import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import random
import numpy as np

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="GS KR Sales Dashboard (Demo)")

# --- 담당자 리스트 ---
BDR_NAMES = ['Sohee (Blair) Kim', 'Soorim Yu', 'Gyeol Jang', 'Minyoung Kim', 'Hyewon Han','Minjeong Jang','Chanwoo Bae']
AE_NAMES = ['Seheon Bok', 'Buheon Shin', 'Ethan Lee', 'Iseul Lee', 'Samin Park', 'Haran Bae']
ALL_PICS = ['All'] + sorted(BDR_NAMES + AE_NAMES)

# --- Deal Stage ID 매핑 (실제 값 대신 이름 사용) ---
DEAL_STAGES = [
    'Proposal Sent & Service Validation',
    'Contract Sent',
    'Payment Complete',
    'Meeting Booked',
    'Meeting Done',
    'Initial Contact',
    'Response Received',
    'Dropped',
    'Price Negotiation',
    'Contract Signed',
    'Closed Lost',
    'New',
    'Lost',
    'Cancel'
]

# '계약 성사' 및 '실패' 기준
won_stages = ['Contract Signed', 'Payment Complete']
lost_stages = ['Closed Lost', 'Dropped', 'Lost', 'Cancel']

# --- [수정] 가상 데이터 생성 함수 ---
@st.cache_data(ttl=3600, show_spinner="데모 데이터 생성 중...")
def create_mock_data(num_deals=500):
    """HubSpot API 대신 사용할 가상 데이터를 생성합니다."""
    korea_tz = pytz.timezone('Asia/Seoul')
    
    data = []
    for i in range(num_deals):
        # --- 기본 정보 생성 ---
        company = f"고객사_{i+1}"
        deal_name = f"{company} 신규 솔루션 도입"
        amount = random.randint(5000, 150000)
        deal_owner = random.choice(AE_NAMES)
        bdr = random.choice(BDR_NAMES + ['Unassigned'])

        # --- 날짜 생성 (논리적 순서 유지) ---
        create_date = datetime.now(korea_tz) - timedelta(days=random.randint(1, 730))
        
        # 날짜 진행 여부 랜덤 결정
        has_meeting_booked = random.random() > 0.1 # 90%
        has_meeting_done = has_meeting_booked and random.random() > 0.2 # 80%
        has_contract_sent = has_meeting_done and random.random() > 0.3 # 70%
        has_contract_signed = has_contract_sent and random.random() > 0.4 # 60%
        has_payment_complete = has_contract_signed and random.random() > 0.5 # 50%
        is_lost = not has_contract_signed and random.random() > 0.8 # 20%
        
        meeting_booked_date = create_date + timedelta(days=random.randint(1, 10)) if has_meeting_booked else pd.NaT
        meeting_done_date = meeting_booked_date + timedelta(days=random.randint(1, 5)) if has_meeting_done and pd.notna(meeting_booked_date) else pd.NaT
        contract_sent_date = meeting_done_date + timedelta(days=random.randint(5, 20)) if has_contract_sent and pd.notna(meeting_done_date) else pd.NaT
        contract_signed_date = contract_sent_date + timedelta(days=random.randint(1, 15)) if has_contract_signed and pd.notna(contract_sent_date) else pd.NaT
        payment_complete_date = contract_signed_date + timedelta(days=random.randint(1, 7)) if has_payment_complete and pd.notna(contract_signed_date) else pd.NaT
        
        # --- 최종 상태 및 날짜 결정 ---
        last_modified_date = create_date
        all_dates = [d for d in [meeting_booked_date, meeting_done_date, contract_sent_date, contract_signed_date, payment_complete_date] if pd.notna(d)]
        if all_dates:
            last_modified_date = max(all_dates)

        close_date = pd.NaT
        deal_stage = 'Initial Contact'
        if has_payment_complete:
            deal_stage = 'Payment Complete'
            close_date = payment_complete_date
        elif has_contract_signed:
            deal_stage = 'Contract Signed'
            close_date = contract_signed_date
        elif has_contract_sent:
            deal_stage = 'Contract Sent'
        elif has_meeting_done:
            deal_stage = 'Meeting Done'
        elif has_meeting_booked:
            deal_stage = 'Meeting Booked'
        
        if is_lost and not has_contract_signed:
            deal_stage = random.choice(lost_stages)
            close_date = last_modified_date + timedelta(days=random.randint(1, 10))
            last_modified_date = close_date

        # --- 실패/드랍 사유 ---
        failure_reason = "가격 문제" if deal_stage == 'Closed Lost' else None
        dropped_reason = "경쟁사 선택" if deal_stage == 'Dropped' else None
        remark = "담당자 연락 두절" if dropped_reason else None

        # --- 데이터 딕셔너리 추가 ---
        data.append({
            'Deal name': deal_name,
            'Deal Stage': deal_stage,
            'Amount': amount,
            'Create Date': create_date,
            'Close Date': close_date,
            'Last Modified Date': last_modified_date,
            'Deal owner': deal_owner,
            'BDR': bdr,
            'Failure Reason': failure_reason,
            'Dropped Reason': dropped_reason,
            'Dropped Reason (Remark)': remark,
            'Expected Closing Date': create_date + timedelta(days=random.randint(30, 120)),
            'Date Entered Stage': last_modified_date - timedelta(days=random.randint(1, 20)),
            'Contract Sent Date': contract_sent_date,
            'Contract Signed Date': contract_signed_date,
            'Payment Complete Date': payment_complete_date,
            'Meeting Booked Date': meeting_booked_date,
            'Meeting Done Date': meeting_done_date,
        })
        
    df = pd.DataFrame(data)
    
    # 날짜 컬럼 타입 변환 (오류 무시)
    date_cols = [
        'Create Date', 'Close Date', 'Last Modified Date', 'Expected Closing Date',
        'Date Entered Stage', 'Contract Sent Date', 'Contract Signed Date',
        'Payment Complete Date', 'Meeting Booked Date', 'Meeting Done Date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        if df[col].notna().any():
            df[col] = df[col].dt.tz_convert('Asia/Seoul')

    # Effective Close Date 생성
    if 'Expected Closing Date' in df.columns and 'Close Date' in df.columns:
        df['Effective Close Date'] = df['Expected Closing Date'].fillna(df['Close Date'])
    elif 'Close Date' in df.columns:
        df['Effective Close Date'] = df['Close Date']
    else:
        df['Effective Close Date'] = pd.NaT
        
    return df

# --- UI 및 대시보드 시작 ---
st.title("🎯 Sales Dashboard (Demo Version)")
st.markdown("가상 데이터를 기반으로 **성장 전략**을 수립합니다.")

# [수정] 가상 데이터 로딩 함수 호출
df = create_mock_data()

if df is None or df.empty:
    st.warning("분석할 데이터가 없거나 로딩에 실패했습니다."); st.stop()

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.success("데모 데이터 로딩 완료!")
    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 DEMO DEAL LIST",
        data=csv_data,
        file_name=f"demo_deals_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    sales_quota = st.number_input("분기/월별 Sales Quota (목표 매출, USD) 입력", min_value=0, value=500000, step=10000)
    st.markdown("---")
    filter_type = st.radio(
        "**날짜 필터 기준 선택**",
        ('생성일 기준 (Create Date)', '예상/확정 마감일 기준', '최종 수정일 기준 (Last Modified Date)'),
        help="**생성일 기준:** 특정 기간에 생성된 딜 분석\n\n**예상/확정 마감일 기준:** 특정 기간에 마감될 딜 분석\n\n**최종 수정일 기준:** 특정 기간에 업데이트된 딜 분석"
    )
    if filter_type == '생성일 기준 (Create Date)': filter_col = 'Create Date'
    elif filter_type == '예상/확정 마감일 기준': filter_col = 'Effective Close Date'
    else: filter_col = 'Last Modified Date'
    
    if not df[filter_col].isna().all():
        min_date_val = df[filter_col].min()
        max_date_val = df[filter_col].max()
        if pd.notna(min_date_val) and pd.notna(max_date_val):
            min_date, max_date = min_date_val.date(), max_date_val.date()
            date_range = st.date_input("분석할 날짜 범위 선택", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            st.error(f"'{filter_col}'에 유효한 날짜 데이터가 없어 범위를 설정할 수 없습니다."); st.stop()
    else:
        st.error(f"'{filter_col}' 데이터가 없어 필터를 설정할 수 없습니다."); st.stop()

# --- 데이터 필터링 ---
korea_tz = pytz.timezone('Asia/Seoul')
start_date = korea_tz.localize(datetime.combine(date_range[0], datetime.min.time())) if len(date_range) > 0 else korea_tz.localize(datetime.min)
end_date = korea_tz.localize(datetime.combine(date_range[1], datetime.max.time())) if len(date_range) > 1 else korea_tz.localize(datetime.max)

base_df = df[df[filter_col].between(start_date, end_date)].copy()

won_deals_df = df[df['Deal Stage'].isin(won_stages)].copy()
signed_in_period = won_deals_df['Contract Signed Date'].between(start_date, end_date)
paid_in_period = won_deals_df['Payment Complete Date'].between(start_date, end_date)
deals_won_in_period = won_deals_df[signed_in_period.fillna(False) | paid_in_period.fillna(False)]

# --- 메인 대시보드 (이하 코드는 원본과 동일) ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 통합 대시보드", "🧑‍💻 담당자별 상세 분석", "⚠️ 기회 & 리스크 관리", "📉 실패/드랍 분석"])

with tab1:
    st.header("팀 전체 현황 요약")
    
    won_deals_total = deals_won_in_period
    
    total_revenue, num_won_deals = won_deals_total['Amount'].sum(), len(won_deals_total)
    avg_deal_value = total_revenue / num_won_deals if num_won_deals > 0 else 0
    
    if not won_deals_total.empty and won_deals_total['Contract Signed Date'].notna().all() and won_deals_total['Create Date'].notna().all():
        valid_cycle_deals = won_deals_total.dropna(subset=['Contract Signed Date', 'Create Date'])
        avg_sales_cycle = (valid_cycle_deals['Contract Signed Date'] - valid_cycle_deals['Create Date']).dt.days.mean()
    else:
        avg_sales_cycle = 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 계약 금액 (USD)", f"${total_revenue:,.0f}")
    col2.metric("계약 성사 건수", f"{num_won_deals:,} 건")
    col3.metric("평균 계약 금액 (USD)", f"${avg_deal_value:,.0f}")
    col4.metric("평균 영업 사이클", f"{avg_sales_cycle:.1f} 일")

    st.markdown("---")
    st.subheader("파이프라인 효율성 분석")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**단계별 전환율 (Funnel)**")
        funnel_stages_map = {
            'Initial Contact': 'Create Date', 
            'Meeting Booked': "Meeting Booked Date",
            'Meeting Done': "Meeting Done Date",
            'Contract Sent': "Contract Sent Date",
            'Contract Signed': "Contract Signed Date"
        }
        funnel_data = []
        funnel_data.append({'Stage': 'Initial Contact', 'Count': base_df['Create Date'].notna().sum()})
        for stage, date_col in funnel_stages_map.items():
            if stage != 'Initial Contact' and base_df.get(date_col) is not None:
                count = base_df[date_col].notna().sum()
                funnel_data.append({'Stage': stage, 'Count': count})

        if len(funnel_data) > 1:
            funnel_df = pd.DataFrame(funnel_data).query("Count > 0")
            fig_funnel = go.Figure(go.Funnel(y=funnel_df['Stage'], x=funnel_df['Count'], textposition="inside", textinfo="value+percent initial"))
            st.plotly_chart(fig_funnel, use_container_width=True)
        else:
            st.warning("Funnel 차트를 그리기에 데이터(날짜 컬럼)가 부족합니다.")

    with col2:
        st.markdown("**단계별 평균 소요 시간 (일)**")
        stage_transitions = [
            {'label': 'Create → Meeting Booked', 'start': 'Create Date', 'end': 'Meeting Booked Date'},
            {'label': 'Booked → Done', 'start': 'Meeting Booked Date', 'end': 'Meeting Done Date'},
            {'label': 'Done → Contract Sent', 'start': 'Meeting Done Date', 'end': 'Contract Sent Date'},
            {'label': 'Sent → Signed', 'start': 'Contract Sent Date', 'end': 'Contract Signed Date'}
        ]
        avg_times = []
        for trans in stage_transitions:
            start_col, end_col = trans['start'], trans['end']
            if base_df.get(start_col) is not None and base_df.get(end_col) is not None:
                valid_deals = base_df.dropna(subset=[start_col, end_col])
                if not valid_deals.empty:
                    time_diff = (valid_deals[end_col] - valid_deals[start_col]).dt.days
                    avg_days = time_diff[time_diff >= 0].mean()
                    if pd.notna(avg_days): avg_times.append({'Transition': trans['label'], 'Avg Days': avg_days})
        if avg_times:
            time_df = pd.DataFrame(avg_times)
            fig_time = px.bar(time_df, x='Avg Days', y='Transition', orientation='h', text='Avg Days')
            fig_time.update_traces(texttemplate='%{text:.1f}일', textposition='auto')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.warning("단계별 소요 시간을 계산할 데이터(날짜 컬럼)가 부족합니다.")

with tab2:
    selected_pic = st.selectbox("분석할 담당자를 선택하세요.", ALL_PICS)
    st.header(f"'{selected_pic}' 상세 분석")

    if selected_pic == 'All':
        st.subheader("AE Leaderboard")
        ae_base_df = base_df[base_df['Deal owner'].isin(AE_NAMES)]
        if not ae_base_df.empty:
            ae_won_stats = deals_won_in_period[deals_won_in_period['Deal owner'].isin(AE_NAMES)]\
                .groupby('Deal owner')\
                .agg(
                    Deals_Won=('Deal name', 'count'),
                    Total_Revenue=('Amount', 'sum')
                ).reset_index()
            all_ae_df = pd.DataFrame(AE_NAMES, columns=['Deal owner'])
            ae_stats = pd.merge(all_ae_df, ae_won_stats, on='Deal owner', how='left').fillna(0)
            ae_stats = ae_stats.sort_values(by='Total_Revenue', ascending=False)
            st.dataframe(ae_stats.style.format({'Total_Revenue': '${:,.0f}','Deals_Won': '{:,}'}), use_container_width=True, hide_index=True)

        st.subheader("BDR Leaderboard")
        bdr_deals_mask = base_df['BDR'].isin(BDR_NAMES) | base_df['Deal owner'].isin(BDR_NAMES)
        all_bdr_deals = base_df[bdr_deals_mask]
        if not all_bdr_deals.empty:
            bdr_performance = []
            for name in BDR_NAMES:
                person_deals = all_bdr_deals[(all_bdr_deals['BDR'] == name) | (all_bdr_deals['Deal owner'] == name)]
                if not person_deals.empty:
                    initial_contacts = person_deals[person_deals['Deal Stage'] == 'Initial Contact'].shape[0]
                    meetings_booked = person_deals[person_deals['Deal Stage'] == 'Meeting Booked'].shape[0]
                    conversion_rate = meetings_booked / initial_contacts if initial_contacts > 0 else 0.0
                    bdr_performance.append({
                        'BDR': name, 'Initial Contacts': initial_contacts,
                        'Meetings Booked (KPI)': meetings_booked, 'Conversion Rate': conversion_rate
                    })
            if bdr_performance:
                bdr_stats = pd.DataFrame(bdr_performance).sort_values(by='Meetings Booked (KPI)', ascending=False)
                st.dataframe(bdr_stats.style.format({'Conversion Rate': '{:.2%}', 'Initial Contacts': '{:,}', 'Meetings Booked (KPI)': '{:,}'}), use_container_width=True, hide_index=True)
    
    else:
        # 개인별 상세 분석 기능
        if selected_pic in BDR_NAMES:
            filtered_df = base_df[(base_df['BDR'] == selected_pic) | (base_df['Deal owner'] == selected_pic)]
        else: # AE
            filtered_df = base_df[base_df['Deal owner'] == selected_pic]
        
        if filtered_df.empty:
            st.warning("선택된 담당자의 데이터가 없습니다.")
        else:
            won_deals_pic = deals_won_in_period[(deals_won_in_period['Deal owner'] == selected_pic) | (deals_won_in_period['BDR'] == selected_pic)]
            open_deals_pic = filtered_df[~filtered_df['Deal Stage'].isin(won_stages + lost_stages)]
            
            st.subheader(f"{selected_pic} 성과 요약")
            
            if selected_pic in AE_NAMES:
                meetings_done = filtered_df['Meeting Done Date'].notna().sum()
                deals_won = len(won_deals_pic)
                total_revenue_pic = won_deals_pic['Amount'].sum()
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("진행 중인 딜", f"{len(open_deals_pic):,} 건")
                c2.metric("미팅 완료", f"{meetings_done:,} 건")
                c3.metric("계약 성사 (기간 내)", f"{deals_won:,} 건")
                c4.metric("총 계약 금액 (기간 내)", f"${total_revenue_pic:,.0f}")
            
            if selected_pic in BDR_NAMES:
                initial_contacts = filtered_df[filtered_df['Deal Stage'] == 'Initial Contact'].shape[0]
                meetings_booked = filtered_df[filtered_df['Deal Stage'] == 'Meeting Booked'].shape[0]
                conversion_rate = meetings_booked / initial_contacts if initial_contacts > 0 else 0.0

                c1, c2, c3 = st.columns(3)
                c1.metric("Initial Contacts", f"{initial_contacts:,} 건")
                c2.metric("Meetings Booked", f"{meetings_booked:,} 건")
                c3.metric("전환율", f"{conversion_rate:.2%}")

            st.markdown("---")
            st.subheader("진행 중인 딜 목록")
            if not open_deals_pic.empty:
                st.dataframe(open_deals_pic[['Deal name', 'Amount', 'Deal Stage', 'Effective Close Date']], use_container_width=True)
            else:
                st.info("현재 진행 중인 딜이 없습니다.")
            
            st.subheader("기간 내 성사시킨 딜 목록")
            if not won_deals_pic.empty:
                st.dataframe(won_deals_pic[['Deal name', 'Amount', 'Contract Signed Date', 'Payment Complete Date']], use_container_width=True)
            else:
                st.info("선택된 기간에 성사시킨 딜이 없습니다.")

with tab3:
    st.header("주요 딜 관리 및 리스크 분석")
    st.subheader("🎯 Next Focus (마감 임박 딜)")
    focus_days = st.selectbox("집중할 기간(일)을 선택하세요:", (30, 60, 90), index=0)
    today = datetime.now(korea_tz)
    days_later = today + timedelta(days=focus_days)
    all_open_deals = df[~df['Deal Stage'].isin(won_stages + lost_stages)]
    focus_deals = all_open_deals[
        (all_open_deals.get('Effective Close Date').notna()) &
        (all_open_deals.get('Effective Close Date') >= today) &
        (all_open_deals.get('Effective Close Date') <= days_later)
    ].sort_values('Amount', ascending=False)
    if not focus_deals.empty:
        focus_deals['Days to Close'] = (focus_deals['Effective Close Date'] - today).dt.days
        st.dataframe(focus_deals[['Deal name', 'Deal owner', 'Amount', 'Effective Close Date', 'Days to Close']].style.format({'Amount': '${:,.0f}'}), use_container_width=True)
    else:
        st.info(f"향후 {focus_days}일 내에 마감될 것으로 예상되는 딜이 없습니다.")

    st.markdown("---")
    st.subheader("👀 장기 체류 딜 (Stale Deals) 관리")
    open_deals_base = base_df[~base_df['Deal Stage'].isin(won_stages + lost_stages)]
    stale_threshold = st.slider("며칠 이상 같은 단계에 머물면 '장기 체류'로 볼까요?", 7, 90, 30)
    
    if 'Date Entered Stage' in open_deals_base.columns:
        open_deals_stale = open_deals_base.copy().dropna(subset=['Date Entered Stage'])
        if pd.api.types.is_datetime64_any_dtype(open_deals_stale['Date Entered Stage']):
            open_deals_stale['Days in Stage'] = (today - open_deals_stale['Date Entered Stage']).dt.days
            stale_deals_df = open_deals_stale[open_deals_stale['Days in Stage'] > stale_threshold]
            if not stale_deals_df.empty:
                st.warning(f"{stale_threshold}일 이상 같은 단계에 머물러 있는 '주의'가 필요한 딜 목록입니다.")
                st.dataframe(stale_deals_df[['Deal name', 'Deal owner', 'Deal Stage', 'Amount', 'Days in Stage']].sort_values('Days in Stage', ascending=False).style.format({'Amount': '${:,.0f}', 'Days in Stage': '{:.0f}일'}), use_container_width=True)
            else:
                st.success("선택된 조건에 해당하는 장기 체류 딜이 없습니다. 👍")
        else:
            st.error("'Date Entered Stage' 컬럼이 날짜 형식이 아니어서 '장기 체류 딜'을 계산할 수 없습니다.")
    else:
        st.warning("'장기 체류 딜' 분석을 위해서는 HubSpot에서 'hs_v2_date_entered_current_stage' 속성을 포함하여 가져와야 합니다.")

with tab4:
    st.header("실패 및 드랍 딜 회고")
    lost_dropped_deals = df[df['Deal Stage'].isin(lost_stages)]
    if not lost_dropped_deals.empty:
        sorted_deals = lost_dropped_deals.sort_values(by='Last Modified Date', ascending=False)
        display_cols = ['Deal name', 'Deal owner', 'Amount', 'Deal Stage', 'Last Modified Date', 'Failure Reason', 'Dropped Reason', 'Dropped Reason (Remark)']
        existing_display_cols = [col for col in display_cols if col in sorted_deals.columns]
        st.dataframe(sorted_deals[existing_display_cols].style.format({'Amount': '${:,.0f}'}), use_container_width=True)
    else:
        st.info("'Closed Lost', 'Dropped', 'Lost', 'Cancel' 상태의 딜이 없습니다.")
