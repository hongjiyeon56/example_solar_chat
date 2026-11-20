import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 한글 폰트 CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CSV 읽기 함수 (인코딩 자동 처리)
# ─────────────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ─────────────────────────────────────────────
# 열 이름 표준화 함수
# ─────────────────────────────────────────────
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
                df[c].astype(str).str.replace(",", "").
                str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ─────────────────────────────────────────────
# 학교명 추출 함수
# ─────────────────────────────────────────────
def get_school_name(filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    parts = name.split("_")
    return parts[0] if parts else name

# ─────────────────────────────────────────────
# CSV 다운로드 헬퍼
# ─────────────────────────────────────────────
def make_download_button(label, df, key):
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv_data, file_name=f"{key}.csv",
                       mime="text/csv", key=key)

# ─────────────────────────────────────────────
# 추세선 추가 헬퍼 (np.polyfit 기반)
# ─────────────────────────────────────────────
def add_trendline(fig, x, y, color="red", dash="dash"):
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return
    xp, yp = valid.iloc[:, 0], valid.iloc[:, 1]
    coef = np.polyfit(xp, yp, 1)
    xs = np.linspace(xp.min(), xp.max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=coef[0] * xs + coef[1],
        mode="lines", line=dict(color=color, dash=dash),
        name=f"추세선 (r={np.corrcoef(xp, yp)[0,1]:.2f})"
    ))

# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.sidebar.title("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

env_df = pd.DataFrame()
growth_df = pd.DataFrame()
school_name = "우리 학교"

if env_file is not None:
    env_df = read_csv(env_file)
    env_df = std_cols(env_df)
    school_name = get_school_name(env_file.name)

if growth_file is not None:
    growth_df = read_csv(growth_file)
    growth_df = std_cols(growth_df)
    if school_name == "우리 학교":
        school_name = get_school_name(growth_file.name)

# 사이드바 측정 항목 선택
growth_items = []
if not growth_df.empty:
    growth_items = [
        c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
        if c in growth_df.columns
    ]

selected_items = st.sidebar.multiselect(
    "🌿 생육 측정 항목 선택 (분석 대상)",
    options=growth_items,
    default=growth_items
)

# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ═══════════════════════════════════════════
# Tab 1: 실험 개요
# ═══════════════════════════════════════════
with tab1:
    try:
        st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

        col_bg, col_null = st.columns([3, 1])
        with col_bg:
            st.markdown(f"""
**연구 배경 및 목적**

극지(寒地) 식물은 낮은 온도와 짧은 생장기에 적응한 독특한 생육 전략을 보입니다.
{school_name}에서 재배한 극지식물 개체들을 일정 기간 관찰하여,
**개체마다 생장 속도, 잎 수, 지상부·지하부 길이가 어떻게 다른지** 비교하고
어떤 개체가 특히 잘 자라는지 확인하는 것이 이 분석의 목적입니다.

- **표준편차**: 값들이 평균에서 얼마나 넓게 퍼져 있는지를 나타내는 값이에요. 클수록 개체 간 차이가 큽니다.
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값이에요. 평균이 다른 집단끼리도 상대적 산포를 비교할 수 있습니다.
- **상관계수**: 두 변수가 함께 움직이는 정도를 −1에서 +1 사이로 나타낸 값이에요. +1에 가까울수록 함께 증가하고, −1에 가까울수록 반대 방향으로 움직여요.
            """)

        # 데이터 요약 표
        st.subheader("📋 데이터 요약")

        summary_rows = []
        if not env_df.empty:
            times = env_df["time"].dropna()
            if len(times) > 0:
                summary_rows.append({
                    "항목": "환경 데이터",
                    "개체(기록) 수": f"{len(env_df)}건",
                    "측정 기간": f"{times.min().strftime('%Y-%m-%d')} ~ {times.max().strftime('%Y-%m-%d')}",
                    "측정 항목": ", ".join(
                        [c for c in ["temperature", "humidity", "ph", "ec"]
                         if c in env_df.columns]
                    )
                })
        if not growth_df.empty:
            summary_rows.append({
                "항목": "생육 결과 데이터",
                "개체(기록) 수": f"{len(growth_df)}개체",
                "측정 기간": "1회 측정 (단일 시점)",
                "측정 항목": ", ".join(
                    [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                     if c in growth_df.columns]
                )
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")
        else:
            st.info("⚠️ CSV 파일을 업로드하면 데이터 요약이 표시됩니다.")

        st.subheader("📊 주요 지표")

        m1, m2, m3, m4 = st.columns(4)

        if not growth_df.empty and "생중량" in growth_df.columns:
            gw = growth_df["생중량"].dropna()
            if len(gw) > 0:
                m1.metric("👾 총 개체수", f"{len(growth_df)}개체")
                m4.metric("⚖️ 평균 생중량", f"{gw.mean():.2f} g")
            else:
                m1.metric("👾 총 개체수", f"{len(growth_df)}개체")
                m4.metric("⚖️ 평균 생중량", "데이터 없음")
        else:
            m1.metric("👾 총 개체수", "CSV 필요")
            m4.metric("⚖️ 평균 생중량", "CSV 필요")

        if not env_df.empty and "temperature" in env_df.columns:
            t = env_df["temperature"].dropna()
            if len(t) > 0:
                m2.metric("🌡️ 평균 온도", f"{t.mean():.1f} °C")
            else:
                m2.metric("🌡️ 평균 온도", "데이터 없음")
        else:
            m2.metric("🌡️ 평균 온도", "CSV 필요")

        if not env_df.empty and "humidity" in env_df.columns:
            h = env_df["humidity"].dropna()
            if len(h) > 0:
                m3.metric("💧 평균 습도", f"{h.mean():.1f} %")
            else:
                m3.metric("💧 평균 습도", "데이터 없음")
        else:
            m3.metric("💧 평균 습도", "CSV 필요")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════
# Tab 2: 환경 데이터
# ═══════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("⚠️ 환경 CSV를 사이드바에 업로드하면 환경 데이터가 표시됩니다.")
        else:
            st.subheader("🌡️ 환경 데이터 요약")

            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                        if c in env_df.columns]
            if env_cols:
                summary = env_df[env_cols].dropna().agg(["mean", "min", "max"])
                summary.index = ["평균", "최소", "최대"]
                st.dataframe(summary.round(2), width="stretch")
            else:
                st.info("⚠️ 환경 데이터에 인식할 수 있는 측정 열이 없어요.")

            # 일별 시계열
            if "time" in env_df.columns and env_cols:
                env_df_sorted = env_df.dropna(subset=["time"]).sort_values("time")
                env_df_sorted["day"] = env_df_sorted["time"].dt.date

                st.subheader("📈 일별 환경 변화 (일별 평균)")

                ts_tab1, ts_tab2, ts_tab3 = st.tabs(
                    ["온도 변화", "습도 변화", "pH·EC 변화"]
                )

                with ts_tab1:
                    if "temperature" in env_df.columns:
                        daily_t = env_df_sorted.groupby("day")["temperature"].mean().dropna()
                        if len(daily_t) > 1:
                            fig_t = go.Figure()
                            fig_t.add_trace(go.Scatter(
                                x=daily_t.index.astype(str),
                                y=daily_t.values,
                                mode="lines+markers",
                                line=dict(color="#ff6b6b", width=2),
                                marker=dict(size=5),
                                name="평균 온도"
                            ))
                            fig_t.update_layout(
                                title=dict(text="🌡️ 일별 평균 온도 변화", x=0.5,
                                           font=dict(size=13)),
                                xaxis_title="날짜",
                                yaxis_title="온도 (°C)",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                height=350
                            )
                            st.plotly_chart(fig_t, width="stretch")
                        else:
                            st.info("⚠️ 일별 온도 데이터가 충분하지 않아요.")
                    else:
                        st.info("⚠️ 환경 데이터에 온도 열이 없어요.")

                with ts_tab2:
                    if "humidity" in env_df.columns:
                        daily_h = env_df_sorted.groupby("day")["humidity"].mean().dropna()
                        if len(daily_h) > 1:
                            fig_h = go.Figure()
                            fig_h.add_trace(go.Scatter(
                                x=daily_h.index.astype(str),
                                y=daily_h.values,
                                mode="lines+markers",
                                line=dict(color="#4ecdc4", width=2),
                                marker=dict(size=5),
                                name="평균 습도"
                            ))
                            fig_h.update_layout(
                                title=dict(text="💧 일별 평균 습도 변화", x=0.5,
                                           font=dict(size=13)),
                                xaxis_title="날짜",
                                yaxis_title="습도 (%)",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                height=350
                            )
                            st.plotly_chart(fig_h, width="stretch")
                        else:
                            st.info("⚠️ 일별 습도 데이터가 충분하지 않아요.")
                    else:
                        st.info("⚠️ 환경 데이터에 습도 열이 없어요.")

                with ts_tab3:
                    ph_ec_cols = [c for c in ["ph", "ec"] if c in env_df.columns]
                    if ph_ec_cols:
                        daily_other = env_df_sorted.groupby("day")[ph_ec_cols].mean().dropna()
                        if len(daily_other) > 1:
                            fig_oe = make_subplots(
                                rows=len(ph_ec_cols), cols=1,
                                shared_xaxes=True,
                                subplot_titles=["pH 변화", "EC 변화"],
                                vertical_spacing=0.15
                            )
                            colors = {"ph": "#9b59b6", "ec": "#f39c12"}
                            for i, col in enumerate(ph_ec_cols):
                                fig_oe.add_trace(go.Scatter(
                                    x=daily_other.index.astype(str),
                                    y=daily_other[col],
                                    mode="lines+markers",
                                    line=dict(color=colors[col], width=2),
                                    marker=dict(size=4),
                                    name=col
                                ), row=i+1, col=1)
                            fig_oe.update_layout(
                                title=dict(text="🧪 일별 pH·EC 변화", x=0.5,
                                           font=dict(size=13)),
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                height=350 * len(ph_ec_cols),
                                showlegend=False
                            )
                            st.plotly_chart(fig_oe, width="stretch")
                        else:
                            st.info("⚠️ 일별 pH·EC 데이터가 충분하지 않아요.")
                    else:
                        st.info("⚠️ 환경 데이터에 pH·EC 열이 없어요.")

            # 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                make_download_button("환경 데이터 CSV 다운로드", env_df, "env_download")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════
# Tab 3: 생육 결과
# ═══════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info("⚠️ 생육 결과 CSV를 사이드바에 업로드하면 생육 분석 결과가 표시됩니다.")
        else:
            # ── 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")

            k1, k2, k3, k4 = st.columns(4)

            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                if len(gw) > 0:
                    mean_gw = gw.mean()
                    max_idx = gw.idxmax()
                    min_idx = gw.idxmin()
                    k1.metric("📈 평균 생중량", f"{mean_gw:.2f} g")
                    k2.metric("🏆 최대 생중량",
                              f"{gw[max_idx]:.2f} g (개체 {growth_df.loc[max_idx, '개체번호']})")
                    k3.metric("📉 최소 생중량",
                              f"{gw[min_idx]:.2f} g (개체 {growth_df.loc[min_idx, '개체번호']})")
                    std_gw = gw.std()
                    cv_gw = (std_gw / mean_gw * 100) if mean_gw > 0 else 0
                    k4.metric("📊 CV(변동계수)", f"{cv_gw:.1f}%")
                else:
                    k1.metric("📈 평균 생중량", "데이터 없음")
                    k2.metric("🏆 최대 생중량", "데이터 없음")
                    k3.metric("📉 최소 생중량", "데이터 없음")
                    k4.metric("📊 CV(변동계수)", "데이터 없음")
            else:
                st.info("⚠️ 생육 데이터에 '생중량' 열이 없어서 핵심 지표를 계산할 수 없어요.")
                k1.metric("📈 평균 생중량", "열 없음")
                k2.metric("🏆 최대 생중량", "열 없음")
                k3.metric("📉 최소 생중량", "열 없음")
                k4.metric("📊 CV(변동계수)", "열 없음")

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📊 개체별 생육 비교")

            bar_cols = {
                "생중량": ("#2ecc71", "생중량 (g)"),
                "잎 수": ("#f39c12", "잎 수 (장)"),
                "지상부 길이": ("#3498db", "지상부 길이 (mm)"),
                "지하부 길이": ("#e74c3c", "지하부 길이 (mm)"),
            }

            available_bars = [c for c in bar_cols if c in growth_df.columns]
            if not available_bars:
                st.info("⚠️ 생육 데이터에 막대 그래프로 표시할 측정 항목이 없어요.")
            else:
                for col in available_bars:
                    color, ylabel = bar_cols[col]
                    valid = growth_df[["개체번호", col]].dropna()
                    if len(valid) >= 2:
                        avg_val = valid[col].mean()
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=valid["개체번호"].astype(str),
                            y=valid[col],
                            marker_color=color,
                            hovertemplate=f"개체: %{{x}}<br>{ylabel}: %{{y:.1f}}<extra></extra>"
                        ))
                        fig.add_hline(y=avg_val, line_dash="dot",
                                      line_color="gray",
                                      annotation_text=f"평균 {avg_val:.1f}",
                                      annotation_position="top right")
                        fig.update_layout(
                            title=dict(text=f"🌿 개체별 평균 {col}", x=0.5,
                                       font=dict(size=13)),
                            xaxis_title="개체번호",
                            yaxis_title=ylabel,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(t=50),
                            height=320
                        )
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"⚠️ '{col}' 데이터가 부족해 막대 그래프를 표시할 수 없어요.")

            # ── 생중량 분포 (박스플롯) ──
            if "생중량" in growth_df.columns:
                gw_valid = growth_df[["개체번호", "생중량"]].dropna()
                if len(gw_valid) >= 3:
                    st.subheader("📦 생중량 분포 (박스플롯)")

                    mean_val = gw_valid["생중량"].mean()
                    q1 = gw_valid["생중량"].quantile(0.25)
                    q3 = gw_valid["생중량"].quantile(0.75)

                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=gw_valid["생중량"],
                        name="생중량",
                        fillcolor="#2ecc71",
                        marker=dict(color="#27ae60"),
                        boxpoints="outliers",
                        hovertemplate="생중량: %{{y:.2f}} g<extra></extra>"
                    ))
                    fig_box.add_hline(y=mean_val, line_dash="dot",
                                      line_color="red",
                                      annotation_text=f"평균 {mean_val:.2f} g",
                                      annotation_position="top right")
                    fig_box.add_hline(y=q1, line_dash="dash",
                                      line_color="orange",
                                      annotation_text=f"Q1 {q1:.2f} g",
                                      annotation_position="bottom right")
                    fig_box.add_hline(y=q3, line_dash="dash",
                                      line_color="orange",
                                      annotation_text=f"Q3 {q3:.2f} g",
                                      annotation_position="top right")
                    fig_box.update_layout(
                        title=dict(text="📦 개체별 생중량 분포", x=0.5,
                                   font=dict(size=13)),
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=360,
                        margin=dict(t=50)
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    # ── 상위 25% / 하위 25% 비교 ──
                    st.subheader("🏅 생중량 상위 25% vs 하위 25% 개체 비교")

                    if len(gw_valid) >= 4:
                        sorted_gw = gw_valid.sort_values("생중량", ascending=False).reset_index(drop=True)
                        n = len(sorted_gw)
                        top_n = max(1, int(np.ceil(n * 0.25)))
                        bottom_n = max(1, int(np.ceil(n * 0.25)))

                        top_df = sorted_gw.head(top_n)
                        bottom_df = sorted_gw.tail(bottom_n)

                        c1, c2 = st.columns(2)

                        with c1:
                            st.markdown(f"**🔝 상위 25% (최상위 {top_n}개체)**")
                            st.dataframe(
                                top_df[["개체번호", "생중량"]],
                                width="stretch"
                            )
                            top_mean = top_df["생중량"].mean()
                            st.info(f"상위 25% 평균 생중량: **{top_mean:.2f} g**")

                        with c2:
                            st.markdown(f"**🔻 하위 25% (최하위 {bottom_n}개체)**")
                            st.dataframe(
                                bottom_df[["개체번호", "생중량"]],
                                width="stretch"
                            )
                            bottom_mean = bottom_df["생중량"].mean()
                            st.info(f"하위 25% 평균 생중량: **{bottom_mean:.2f} g**")

                        gap = top_mean - bottom_mean
                        st.success(f"상위-하위 평균 격차: **{gap:.2f} g** "
                                   f"({gap/bottom_mean*100:.1f}% 차이)")
                    else:
                        st.info("⚠️ 개체 수가 너무 적어 상위·하위 25%를 나눌 수 없어요.")
            else:
                st.info("⚠️ 생중량 열이 없어서 박스플롯과 분위수 비교를 표시할 수 없어요.")

            # ── 상관관계 분석 ──
            st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")

            corr_pairs = []
            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                corr_pairs.append(("잎 수", "생중량", "#f39c12"))
            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                corr_pairs.append(("지상부 길이", "생중량", "#3498db"))

            if not corr_pairs:
                st.info("⚠️ 상관계수를 계산할 수 있는 변수 쌍이 없어요.")
            else:
                for x_col, y_col, color in corr_pairs:
                    pair_df = growth_df[[x_col, y_col]].dropna()
                    if len(pair_df) >= 2:
                        r = pair_df[x_col].corr(pair_df[y_col])
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=pair_df[x_col],
                            y=pair_df[y_col],
                            mode="markers",
                            marker=dict(color=color, size=10,
                                        line=dict(width=1, color="white")),
                            name=f"{x_col} vs {y_col}",
                            hovertemplate=f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.2f}}<extra></extra>"
                        ))
                        add_trendline(fig_scatter, pair_df[x_col], pair_df[y_col], color)
                        fig_scatter.update_layout(
                            title=dict(
                                text=f"{x_col} vs {y_col} (상관계수 r = {r:.2f})",
                                x=0.5, font=dict(size=13)
                            ),
                            xaxis_title=x_col,
                            yaxis_title=f"{y_col} (g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            height=380,
                            margin=dict(t=50)
                        )
                        st.plotly_chart(fig_scatter, width="stretch")

                        strength = ("강한" if abs(r) >= 0.7 else
                                    "중간" if abs(r) >= 0.4 else
                                    "약한")
                        direction = "정" if r > 0 else "부"
                        st.markdown(f"""
**💡 해석**: {x_col}과 {y_col}의 상관계수는 **r = {r:.2f}** 로,
{int(abs(r)*100)}% 수준의 **{strength} {direction}적 상관**을 보여요.
                        """)
                    else:
                        st.info(f"⚠️ '{x_col}'과 '{y_col}' 데이터가 부족해 상관계수를 계산할 수 없어요.")

            # ── 원본 테이블 + 다운로드 ──
            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                make_download_button("생육 결과 CSV 다운로드", growth_df, "growth_download")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
