import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── CSV 읽기 (인코딩 이중 시도) ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ───────────────────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") \
                + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ── 학교명 추출 ──────────────────────────────────────────────────────────────
def get_school_name(filename):
    base = filename.replace(".csv", "")
    return base.split("_")[0]

# ── 공통 스타일 적용 헬퍼 ────────────────────────────────────────────────────
def apply_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig

# ── 서술통계량 계산 헬퍼 ─────────────────────────────────────────────────────
def stats_table(df, cols, label_map=None):
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) < 2:
            continue
        row = {
            "항목": label_map.get(c, c) if label_map else c,
            "평균": round(s.mean(), 2),
            "중앙값": round(s.median(), 2),
            "최솟값": round(s.min(), 2),
            "최댓값": round(s.max(), 2),
            "표준편차": round(s.std(), 2),
            "데이터 수": len(s),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ── 결측 구간 탐지 ───────────────────────────────────────────────────────────
def find_missing_intervals(df):
    if "time" not in df.columns or df["time"].isna().all():
        return []
    times = df["time"].dropna().sort_values()
    if len(times) < 2:
        return []
    gaps = []
    for i in range(1, len(times)):
        diff = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 3600
        if diff > 3:
            gaps.append((times.iloc[i - 1], times.iloc[i], round(diff, 1)))
    return gaps

# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 분석 대시보드", layout="wide")
st.title("🌱 극지식물 생육 분석 대시보드")
st.caption("재배 환경 변화와 개체별 생육 특성을 데이터로 비교하고 분석하기")

# ── 사이드바: 파일 업로드 ─────────────────────────────────────────────────────
env_file = st.sidebar.file_uploader("📁 환경 데이터 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("📁 생육 결과 데이터 CSV 업로드", type="csv")

env_school = None
growth_school = None
env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.sidebar.spinner("환경 데이터 로딩 중..."):
        raw_env = read_csv(env_file)
        if not raw_env.empty:
            env_df = std_cols(raw_env)
            env_school = get_school_name(env_file.name)
        else:
            st.sidebar.warning("환경 CSV를 읽을 수 없어요. 인코딩을 확인해 주세요.")

if growth_file is not None:
    with st.sidebar.spinner("생육 데이터 로딩 중..."):
        raw_growth = read_csv(growth_file)
        if not raw_growth.empty:
            growth_df = std_cols(raw_growth)
            growth_school = get_school_name(growth_file.name)
        else:
            st.sidebar.warning("생육 CSV를 읽을 수 없어요. 인코딩을 확인해 주세요.")

# ── 사이드바: 파일 정보 표시 ─────────────────────────────────────────────────
if env_df is not None and not env_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 환경 데이터 정보")
    st.sidebar.info(f"**학교명:** {env_school}")
    st.sidebar.info(f"**데이터 행 수:** {len(env_df):,}행")
    env_cols_display = [c for c in env_df.columns if c != "time"]
    if env_cols_display:
        st.sidebar.info(f"**측정 항목:** {', '.join(env_cols_display)}")
    else:
        st.sidebar.info("측정 항목이 인식되지 않았어요.")

if growth_df is not None and not growth_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 생육 데이터 정보")
    st.sidebar.info(f"**학교명:** {growth_school}")
    st.sidebar.info(f"**총 개체 수:** {growth_df['개체번호'].nunique()}개체")
    growth_cols_display = [c for c in growth_df.columns if c not in ("개체번호", "time")]
    if growth_cols_display:
        st.sidebar.info(f"**생육 측정 항목:** {', '.join(growth_cols_display)}")
    else:
        st.sidebar.info("생육 측정 항목이 인식되지 않았어요.")

st.sidebar.markdown("---")
st.sidebar.caption("CSV 파일명의 앞부분이 학교명으로 자동 표시돼요.")
st.sidebar.caption("인코딩 오류가 있으면 cp949로 다시 읽어요.")

# ── 생육 측정 항목 선택 ──────────────────────────────────────────────────────
growth_cols_all = [
    "잎 수", "지상부 길이", "지하부 길이", "생중량",
    "지상부 질량", "지하부 질량",
]
available_growth = [c for c in growth_cols_all if c in growth_df.columns] \
    if not growth_df.empty else []
selected_growth_col = st.sidebar.selectbox(
    "📊 비교할 생육 항목 선택",
    options=available_growth if available_growth else ["(생육 데이터 없음)"],
    index=0 if available_growth else 0,
)

tabs = st.tabs([
    "🌡️ 극지식물 재배 환경 분석",
    "🌿 극지식물 개체별 생육 비교",
    "📊 생육 특성 간 관계 분석",
])

# ════════════════════════════════════════════════════════════════════════════
# 탭 1: 재배 환경 분석
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 업로드하면 분석이 시작돼요.")
        else:
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                        if c in env_df.columns]

            # ── 상단 metric 4개 ──────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            if "temperature" in env_df.columns:
                t = env_df["temperature"].dropna()
                if len(t) >= 2:
                    col1.metric("🌡️ 평균 온도", f"{t.mean():.1f}℃",
                                delta=f"범위: {t.min():.1f} ~ {t.max():.1f}℃")
            if "humidity" in env_df.columns:
                h = env_df["humidity"].dropna()
                if len(h) >= 2:
                    col2.metric("💧 평균 습도", f"{h.mean():.1f}%",
                                delta=f"범위: {h.min():.1f} ~ {h.max():.1f}%")
            if "ph" in env_df.columns:
                p = env_df["ph"].dropna()
                if len(p) >= 2:
                    col3.metric("🧪 평균 pH", f"{p.mean():.2f}",
                                delta=f"범위: {p.min():.2f} ~ {p.max():.2f}")
            if "ec" in env_df.columns:
                e = env_df["ec"].dropna()
                if len(e) >= 2:
                    col4.metric("⚡ 평균 EC", f"{e.mean():.2f} mS/cm",
                                delta=f"범위: {e.min():.2f} ~ {e.max():.2f} mS/cm")

            st.caption("※ 표준편차: 데이터가 평균 주위에 얼마나 퍼져 있는지를 나타내는 값이에요. 값이 클수록 측정값 변화가 크다는 뜻이에요.")

            st.markdown("---")

            # ── 기간 선택 ─────────────────────────────────────────────────────
            if "time" in env_df.columns and env_df["time"].notna().any():
                tmin = env_df["time"].min()
                tmax = env_df["time"].max()
                st.markdown(f"**측정 기간:** {tmin.strftime('%Y-%m-%d %H:%M')} ~ {tmax.strftime('%Y-%m-%d %H:%M')}")
                period = st.radio(
                    "기간 선택",
                    options=["전체", "일별"],
                    index=0,
                    horizontal=True,
                )
            else:
                period = "전체"

            # ── 환경 변수별 선그래프 (단위 달라서 개별 그래프) ───────────────
            available_env_plot = [c for c in env_cols if c in env_df.columns]
            if available_env_plot:
                st.subheader("📈 환경 측정값의 시간별 변화")
                plot_col1, plot_col2 = st.columns(2)
                plots_per_col = (len(available_env_plot) + 1) // 2

                for idx, col_name in enumerate(available_env_plot):
                    unit_map = {
                        "temperature": "℃",
                        "humidity": "%",
                        "ph": "",
                        "ec": "mS/cm",
                    }
                    unit = unit_map.get(col_name, "")
                    label_kr = {
                        "temperature": "온도",
                        "humidity": "습도",
                        "ph": "pH",
                        "ec": "EC",
                    }.get(col_name, col_name)

                    if idx < plots_per_col:
                        target_col = plot_col1
                    else:
                        target_col = plot_col2

                    with target_col:
                        sub_df = env_df[["time", col_name]].dropna().copy()
                        if len(sub_df) < 2:
                            st.info(f"{label_kr} 데이터가 충분하지 않아 그래프를 표시할 수 없어요.")
                            continue

                        if period == "일별" and "time" in env_df.columns:
                            sub_df["day"] = sub_df["time"].dt.date
                            daily = sub_df.groupby("day")[col_name].mean().reset_index()
                            daily.columns = ["day", col_name]
                            fig = px.line(
                                daily, x="day", y=col_name,
                                labels={"day": "날짜", col_name: f"{label_kr} ({unit})"},
                                markers=True,
                            )
                            fig.update_traces(line_color="#2563EB", line_width=2)
                        else:
                            fig = px.line(
                                sub_df, x="time", y=col_name,
                                labels={"time": "시간", col_name: f"{label_kr} ({unit})"},
                                markers=True,
                            )
                            fig.update_traces(line_color="#2563EB", line_width=1)

                        apply_font(fig)
                        fig.update_layout(height=280, margin=dict(l=40, r=20, t=30, b=40))
                        st.plotly_chart(fig, width="stretch")

            # ── 일별 평균 + 분포 (박스플롯) ───────────────────────────────────
            if period == "일별" and "time" in env_df.columns and available_env_plot:
                st.markdown("---")
                st.subheader("📊 일별 평균 변화와 측정값 분포")
                box_col1, box_col2 = st.columns(2)
                box_count = 0
                for col_name in available_env_plot:
                    label_kr = {
                        "temperature": "온도",
                        "humidity": "습도",
                        "ph": "pH",
                        "ec": "EC",
                    }.get(col_name, col_name)
                    unit = {"temperature": "℃", "humidity": "%", "ph": "", "ec": "mS/cm"}.get(col_name, "")

                    if box_count % 2 == 0:
                        bc = box_col1
                    else:
                        bc = box_col2
                    box_count += 1

                    with bc:
                        plot_df = env_df[["time", col_name]].dropna().copy()
                        if len(plot_df) < 2:
                            st.info(f"{label_kr} 데이터가 부족해요.")
                            continue
                        plot_df["day"] = plot_df["time"].dt.date
                        fig = px.box(
                            plot_df, x="day", y=col_name,
                            labels={"day": "날짜", col_name: f"{label_kr} ({unit})"},
                            color_discrete_sequence=["#7C3AED"],
                        )
                        apply_font(fig)
                        fig.update_layout(height=260, margin=dict(l=40, r=20, t=30, b=40))
                        st.plotly_chart(fig, width="stretch")

            # ── 서술통계량 표 ─────────────────────────────────────────────────
            if available_env_plot:
                st.markdown("---")
                st.subheader("📋 환경 항목별 통계 요약")
                st.caption("평균: 데이터 중심값 / 표준편차: 값이 평균에서 얼마나 흩어져 있는지 / 최솟값·최댓값: 관측된 범위")
                env_label_map = {
                    "temperature": "온도 (℃)",
                    "humidity": "습도 (%)",
                    "ph": "pH",
                    "ec": "EC (mS/cm)",
                }
                env_stats = stats_table(env_df, available_env_plot, env_label_map)
                if not env_stats.empty:
                    st.dataframe(env_stats, width="stretch", hide_index=True)

            # ── 측정 누락 구간 안내 ────────────────────────────────────────────
            missing = find_missing_intervals(env_df)
            if missing:
                st.markdown("---")
                st.warning("⚠️ 센서 데이터가 누락된 시간 구간이 감지되었어요. 아래 구간을 확인해 보세요.")
                miss_df = pd.DataFrame([
                    {
                        "시작 시각": r[0].strftime("%Y-%m-%d %H:%M"),
                        "종료 시각": r[1].strftime("%Y-%m-%d %H:%M"),
                        "공백 길이": f"{r[2]}시간",
                    }
                    for r in missing
                ])
                st.dataframe(miss_df, width="stretch", hide_index=True)
            else:
                st.info("✅ 측정 누락 구간이 감지되지 않았어요. 데이터가 연속적으로 기록되어 있어요.")

            # ── 없는 항목 안내 ─────────────────────────────────────────────────
            for missing_col, label in [("water_temp", "수온"), ("co2", "CO₂")]:
                if missing_col not in env_df.columns:
                    st.caption(f"※ {label} 데이터는 이 파일에 포함되어 있지 않아요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 탭 2: 개체별 생육 비교
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    try:
        if growth_df.empty:
            st.info("🌿 생육 결과 데이터 CSV를 업로드하면 분석이 시작돼요.")
        else:
            # 결측/빈 개체번호 제거
            gdf = growth_df.dropna(subset=["개체번호"]).copy()
            if "개체번호" in gdf.columns:
                gdf["개체번호"] = gdf["개체번호"].astype(str).str.strip()

            # ── 상단 metric ───────────────────────────────────────────────────
            total = gdf["개체번호"].nunique()
            c1, c2, c3, c4 = st.columns(4)
            if "잎 수" in gdf.columns:
                ln = gdf["잎 수"].dropna()
                if len(ln) >= 2:
                    c1.metric("🍃 평균 잎 수", f"{ln.mean():.1f}장",
                              delta=f"범위: {ln.min():.0f} ~ {ln.max():.0f}장")
            if "지상부 길이" in gdf.columns:
                sl = gdf["지상부 길이"].dropna()
                if len(sl) >= 2:
                    c2.metric("🌱 평균 지상부 길이", f"{sl.mean():.1f} mm",
                              delta=f"범위: {sl.min():.0f} ~ {sl.max():.0f} mm")
            if "지하부 길이" in gdf.columns:
                rl = gdf["지하부 길이"].dropna()
                if len(rl) >= 2:
                    c3.metric("🌿 평균 지하부 길이", f"{rl.mean():.1f} mm",
                              delta=f"범위: {rl.min():.0f} ~ {rl.max():.0f} mm")
            if "생중량" in gdf.columns:
                bw = gdf["생중량"].dropna()
                if len(bw) >= 2:
                    c4.metric("⚖️ 평균 생중량", f"{bw.mean():.2f} g",
                              delta=f"범위: {bw.min():.2f} ~ {bw.max():.2f} g")

            st.caption(f"※ 총 {total}개체를 분석했어요.")

            st.markdown("---")

            # ── 선택한 생육 항목 막대그래프 ──────────────────────────────────
            if selected_growth_col and selected_growth_col != "(생육 데이터 없음)" \
                    and selected_growth_col in gdf.columns:
                plot_data = gdf[["개체번호", selected_growth_col]].dropna()
                if len(plot_data) >= 2:
                    mean_val = plot_data[selected_growth_col].mean()
                    # 상위/하위 5개체
                    top5 = plot_data.nlargest(5, selected_growth_col)
                    bot5 = plot_data.nsmallest(5, selected_growth_col)
                    top_ids = set(top5["개체번호"].astype(str))
                    bot_ids = set(bot5["개체번호"].astype(str))
                    # 색상 할당
                    colors = []
                    for _, row in plot_data.iterrows():
                        oid = str(row["개체번호"])
                        if oid in top_ids:
                            colors.append("#DC2626")
                        elif oid in bot_ids:
                            colors.append("#059669")
                        else:
                            colors.append("#93C5FD")

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=plot_data["개체번호"].astype(str),
                        y=plot_data[selected_growth_col],
                        marker_color=colors,
                        hovertemplate=f"개체: %{{x}}<br>{selected_growth_col}: %{{y:.2f}}<extra></extra>",
                    ))
                    fig.add_hline(
                        y=mean_val,
                        line_dash="dash",
                        line_color="#6B7280",
                        annotation_text=f"평균: {mean_val:.2f}",
                        annotation_position="top right",
                    )
                    apply_font(fig)
                    fig.update_layout(
                        xaxis_title="개체번호",
                        yaxis_title=f"{selected_growth_col}",
                        height=420,
                        margin=dict(l=50, r=30, t=40, b=60),
                    )
                    st.plotly_chart(fig, width="stretch")

                    # 상위/하위 표
                    st.markdown("---")
                    st.subheader("🏆 생중량 기준 상위 5개체 / 하위 5개체")
                    if "생중량" in gdf.columns:
                        bw_data = gdf[["개체번호", "생중량"]].dropna()
                        if len(bw_data) >= 5:
                            top_table = bw_data.nlargest(5, "생중량").reset_index(drop=True)
                            bot_table = bw_data.nsmallest(5, "생중량").reset_index(drop=True)
                            top_table.index = top_table.index + 1
                            bot_table.index = bot_table.index + 1
                            top_table.columns = ["개체번호", "생중량 (g)"]
                            bot_table.columns = ["개체번호", "생중량 (g)"]
                            tc1, tc2 = st.columns(2)
                            with tc1:
                                st.markdown("**🔴 생중량이 높은 상위 5개체**")
                                st.dataframe(top_table, width="stretch", hide_index=True)
                            with tc2:
                                st.markdown("**🟢 생중량이 낮은 하위 5개체**")
                                st.dataframe(bot_table, width="stretch", hide_index=True)

                    # 평균 이상/이하 개체 수
                    above = (plot_data[selected_growth_col] > mean_val).sum()
                    below = (plot_data[selected_growth_col] < mean_val).sum()
                    ab1, ab2, ab3 = st.columns(3)
                    ab1.info(f"📈 평균보다 높은 개체: **{above}개체**")
                    ab2.info(f"📉 평균보다 낮은 개체: **{below}개체**")
                    ab3.info(f"📊 전체 평균: **{mean_val:.2f}**")
                else:
                    st.info("선택한 생육 항목 데이터가 부족해요.")

            # ── 분포 (박스플롯) ───────────────────────────────────────────────
            if selected_growth_col and selected_growth_col != "(생육 데이터 없음)" \
                    and selected_growth_col in gdf.columns:
                st.markdown("---")
                st.subheader("📊 생육 항목 분포 (박스플롯)")
                st.caption("박스플롯: 상자 윗부분은 상위 25%, 중앙선은 중앙값, 아랫부분은 하위 25%예요. 수염 밖 점은 극단값이에요.")
                dist_data = gdf[[selected_growth_col]].dropna()
                if len(dist_data) >= 2:
                    fig = px.box(
                        dist_data, y=selected_growth_col,
                        labels={selected_growth_col: selected_growth_col},
                        color_discrete_sequence=["#7C3AED"],
                    )
                    apply_font(fig)
                    fig.update_layout(
                        height=280,
                        margin=dict(l=50, r=30, t=30, b=50),
                    )
                    st.plotly_chart(fig, width="stretch")

            # ── 기술통계량 표 ──────────────────────────────────────────────────
            growth_stat_cols = [c for c in
                                ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                                if c in gdf.columns]
            if growth_stat_cols:
                st.markdown("---")
                st.subheader("📋 생육 항목별 기술통계량")
                growth_label_map = {
                    "잎 수": "잎 수 (장)",
                    "지상부 길이": "지상부 길이 (mm)",
                    "지하부 길이": "지하부 길이 (mm)",
                    "생중량": "생중량 (g)",
                }
                growth_stats = stats_table(gdf, growth_stat_cols, growth_label_map)
                if not growth_stats.empty:
                    st.dataframe(growth_stats, width="stretch", hide_index=True)

            # ── CSV 다운로드 버튼 ─────────────────────────────────────────────
            if not gdf.empty:
                st.markdown("---")
                csv_out = gdf.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 분석 데이터 CSV 다운로드",
                    data=csv_out,
                    file_name="극지식물_생육분석결과.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 탭 3: 상관관계 분석
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    try:
        if growth_df.empty:
            st.info("📊 생육 결과 데이터 CSV를 업로드하면 상관관계 분석이 시작돼요.")
        else:
            gdf3 = growth_df.dropna(subset=["개체번호"]).copy()
            if "개체번호" in gdf3.columns:
                gdf3["개체번호"] = gdf3["개체번호"].astype(str).str.strip()

            num_cols = [
                c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                if c in gdf3.columns
            ]

            if len(num_cols) < 2:
                st.info("상관계수를 계산하려면 생육 측정 항목이 2개 이상 필요해요.")
            else:
                # ── X / Y 축 선택 ────────────────────────────────────────────────
                st.subheader("🔍 두 생육 항목 간 산점도 분석")
                col_x = st.selectbox("X축 항목", options=num_cols, index=0)
                col_y = st.selectbox("Y축 항목", options=num_cols, index=num_cols.index("생중량") if "생중량" in num_cols else 0)

                scatter_data = gdf3[[col_x, col_y]].dropna()
                if len(scatter_data) >= 2:
                    fig = px.scatter(
                        scatter_data,
                        x=col_x,
                        y=col_y,
                        labels={col_x: col_x, col_y: col_y},
                        hover_name=gdf3.loc[scatter_data.index, "개체번호"].astype(str)
                        if "개체번호" in gdf3.columns else None,
                        color_discrete_sequence=["#7C3AED"],
                    )

                    # np.polyfit 추세선 직접 추가
                    x_vals = scatter_data[col_x].values
                    y_vals = scatter_data[col_y].values
                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.array([x_vals.min(), x_vals.max()])
                    y_line = slope * x_line + intercept
                    fig.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="#DC2626", width=2, dash="dash"),
                        name=f"추세선 (기울기: {slope:.4f})",
                    ))

                    # 상관계수
                    corr_val = np.corrcoef(x_vals, y_vals)[0, 1]

                    apply_font(fig)
                    fig.update_layout(
                        xaxis_title=col_x,
                        yaxis_title=col_y,
                        height=460,
                        margin=dict(l=50, r=30, t=40, b=50),
                    )
                    st.plotly_chart(fig, width="stretch")

                    # 상관계수 설명
                    st.markdown("---")
                    st.subheader("📐 상관계수 분석 결과")
                    st.info(
                        f"**상관계수 (r): {corr_val:.4f}**  "
                        f"→ {col_x}와 {col_y} 사이의 선형 관계 강도"
                    )
                    st.caption(
                        "※ 상관계수(r)는 -1에서 1 사이의 값이에요. "
                        "1에 가까울수록 함께 증가하는 경향(양의 상관), "
                        "-1에 가까울수록 반대로 움직이는 경향(음의 상관), "
                        "0에 가까우면 선형 관계가 거의 없다는 뜻이에요."
                    )
                    st.caption(
                        "⚠️ 상관관계는 인과관계를 의미하지 않아요. "
                        "두 값이 함께 변한다고 해서 하나가 다른 하나의 원인이라고 말할 수는 없어요."
                    )
                    st.caption(
                        "⚠️ 이번 생육 데이터는 개체별 단일 측정 결과이므로, "
                        "시간에 따른 성장 속도나 환경 요인이 개별 식물의 생중량을 변화시켰다는 결론은 내릴 수 없어요."
                    )

                else:
                    st.info("선택한 두 항목의 데이터가 충분하지 않아요.")

                # ── 상관계수 히트맵 ─────────────────────────────────────────────
                if len(num_cols) >= 2:
                    st.markdown("---")
                    st.subheader("🗺️ 생육 항목 간 상관계수 히트맵")
                    corr_matrix = gdf3[num_cols].dropna().corr()
                    fig_heat = px.imshow(
                        corr_matrix,
                        text_auto=".3f",
                        color_continuous_scale="RdBu_r",
                        labels=dict(x="항목", y="항목", color="상관계수"),
                    )
                    apply_font(fig_heat)
                    fig_heat.update_layout(
                        height=420,
                        margin=dict(l=50, r=30, t=40, b=50),
                    )
                    st.plotly_chart(fig_heat, width="stretch")

                    # 생중량과의 상관계수 비교
                    if "생중량" in corr_matrix.columns:
                        st.markdown("---")
                        st.subheader("📊 생중량과 다른 생육 항목 간 상관계수 비교")
                        bw_corr = corr_matrix["생중량"].drop("생중량").sort_values(ascending=False)
                        corr_rows = []
                        for c, v in bw_corr.items():
                            strength = "강한 관계" if abs(v) >= 0.7 else \
                                       "중간 관계" if abs(v) >= 0.4 else \
                                       "약한 관계"
                            direction = "양의 상관" if v > 0 else "음의 상관"
                            corr_rows.append({
                                "생육 항목": c,
                                "상관계수 (r)": round(v, 4),
                                "관계 강도": strength,
                                "방향": direction,
                            })
                        corr_table = pd.DataFrame(corr_rows)
                        st.dataframe(corr_table, width="stretch", hide_index=True)
                        st.caption(
                            "※ 상관계수가 높다고 해서 생중량을 결정하는 원인이라고 단정할 수는 없어요. "
                            "식물의 생육은 다양한 요인이 복합적으로 작용해요."
                        )

                # ── 탐구 질문 ───────────────────────────────────────────────────
                st.markdown("---")
                st.subheader("🔎 함께 생각해 볼 탐구 질문")
                st.markdown(
                    "- 잎 수가 많은 식물은 항상 생중량도 클까? 예외인 개체가 있다면 그 이유는 무엇일지 생각해 보세요.\n"
                    "- 뿌리가 긴 식물일수록 생중량이 높을까? 지상부 길이와 지하부 길이의 관계는 어떨까요?\n"
                    "- 개체별 생육 차이가 발생하는 이유는 무엇일까? (빛, 수분, 영양, 유전적 차이 등)"
                )

                # ── 상관 분석 결과 CSV 다운로드 ────────────────────────────────
                if len(num_cols) >= 2:
                    corr_csv = corr_matrix.reset_index().rename(columns={"index": "항목"})
                    csv_out3 = corr_csv.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 상관계수 분석 결과 CSV 다운로드",
                        data=csv_out3,
                        file_name="극지식물_상관관계분석결과.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
