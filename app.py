import re
import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================
# Utils
# =========================

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def months_between(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return (d1 - d0).days / (365.25 / 12.0)

def nearest_on_or_after(idx: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    pos = idx.searchsorted(target, side="left")
    if pos >= len(idx):
        return None
    return idx[pos]

def normalize_bbg_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    line = re.sub(r"\s+", " ", line)
    return line

def parse_overrides(text: str) -> Dict[str, str]:
    """
    Overrides format (one per line):
      BMW GY = BMW.DE
      0700 HK = 0700.HK
    Left side is Bloomberg input (normalized), right side is Yahoo symbol.
    """
    out = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            continue
        left, right = raw.split("=", 1)
        left = normalize_bbg_line(left) or ""
        right = right.strip()
        if left and right:
            out[left] = right
    return out


# =========================
# Bloomberg -> Yahoo conversion (rules + overrides)
# =========================

BBG_EXCH_TO_YAHOO_SUFFIX = {
    # USA
    "US": "",      # AAPL US -> AAPL

    # Europe
    "GY": ".DE",   # BMW GY -> BMW.DE
    "FP": ".PA",   # AIR FP -> AIR.PA
    "LN": ".L",    # VOD LN -> VOD.L
    "SW": ".SW",   # NESN SW -> NESN.SW
    "NA": ".AS",   # ASML NA -> ASML.AS
    "SM": ".MC",   # ITX SM -> ITX.MC
    "IM": ".MI",   # ENEL IM -> ENEL.MI
    "SS": ".ST",   # ERICB SS -> ERICB.ST
    "FH": ".HE",   # NOKIA FH -> NOKIA.HE
    "BB": ".BR",   # ABI BB -> ABI.BR

    # Asia
    "JP": ".T",    # 7203 JP -> 7203.T
    "HK": ".HK",   # 0700 HK -> 0700.HK
    "KS": ".KS",   # 005930 KS -> 005930.KS
    "KQ": ".KQ",   # 035720 KQ -> 035720.KQ
    "TT": ".TW",   # 2330 TT -> 2330.TW
    "SP": ".SI",   # D05 SP -> D05.SI

    # Other common
    "AU": ".AX",   # BHP AU -> BHP.AX
    "CN": ".TO",   # SHOP CN -> SHOP.TO
}

def bbg_to_yahoo(bbg: str, overrides: Dict[str, str]) -> Tuple[str, str]:
    """
    Returns (yahoo_symbol, note)
    note: explanation/warning to show the user
    """
    bbg_norm = normalize_bbg_line(bbg)
    if not bbg_norm:
        return "", "empty"

    if bbg_norm in overrides:
        return overrides[bbg_norm], "override"

    parts = bbg_norm.split(" ")
    if len(parts) < 2:
        # already maybe Yahoo-like or incomplete
        return bbg_norm, "no_exchange_code_assumed_yahoo"

    base = parts[0].strip()
    exch = parts[1].strip().upper()

    # Handle Bloomberg class shares like BRK/B US -> BRK-B
    base_y = base.replace("/", "-")

    # HK: zero-pad numeric tickers to 4 digits
    if exch == "HK":
        if base_y.isdigit():
            base_y = base_y.zfill(4)

    suffix = BBG_EXCH_TO_YAHOO_SUFFIX.get(exch)
    if suffix is None:
        # Unknown exchange mapping -> return base and warn
        return base_y, f"unknown_exchange_code:{exch}"

    return f"{base_y}{suffix}", "rule"


# =========================
# Quotes download (Yahoo)
# =========================

@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1 hour cache
def download_prices(yahoo_symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns adj close prices (or close if adj close missing) in wide format:
    index=date, columns=symbol
    """
    data = yf.download(
        tickers=" ".join(yahoo_symbols),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    # yfinance format:
    # - if one ticker: columns like ["Open","High",...]
    # - if multiple: MultiIndex columns: (Field, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer Adj Close, fallback Close
        if ("Adj Close" in data.columns.get_level_values(0)):
            px = data["Adj Close"].copy()
        else:
            px = data["Close"].copy()
    else:
        # single ticker
        if "Adj Close" in data.columns:
            px = data[["Adj Close"]].rename(columns={"Adj Close": yahoo_symbols[0]})
        else:
            px = data[["Close"]].rename(columns={"Close": yahoo_symbols[0]})

    px = px.dropna(how="all")
    px.index = pd.to_datetime(px.index).normalize()
    # Ensure column order and existence
    for sym in yahoo_symbols:
        if sym not in px.columns:
            px[sym] = np.nan
    px = px[yahoo_symbols]
    return px


# =========================
# Product + backtest (Phoenix worst-of)
# =========================

@dataclass
class PhoenixParams:
    obs_months: int
    term_months: int
    call_barrier: float
    coupon: float
    coupon_barrier: float
    memory: bool
    principal_barrier: float
    leverage_down: float

def backtest_phoenix_worstof(
    px: pd.DataFrame,            # wide df index=date, columns=tickers
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    params: PhoenixParams,
    launch_step: int = 5,
) -> pd.DataFrame:
    px = px.loc[start_date:end_date].copy()
    px = px.dropna(how="all")
    if px.empty:
        raise ValueError("Нет котировок в выбранном диапазоне.")

    # Common calendar: dates where all tickers have prices
    common_mask = px.notna().all(axis=1)
    common_dates = px.index[common_mask]
    if len(common_dates) < 50:
        raise ValueError("Слишком мало общих дат (пересечение по всем тикерам). Проверь тикеры/диапазон.")

    tickers = list(px.columns)
    launch_dates = common_dates[::launch_step]
    rows = []

    for launch in launch_dates:
        s0 = px.loc[launch, tickers].astype(float)
        if (s0 <= 0).any() or (~np.isfinite(s0)).any():
            continue

        # observation schedule
        n_obs = int(params.term_months / params.obs_months)
        obs_dates = []
        for k in range(1, n_obs + 1):
            target = (launch + pd.DateOffset(months=k * params.obs_months)).normalize()
            obs = nearest_on_or_after(common_dates, target)
            if obs is None:
                break
            obs_dates.append(obs)

        if not obs_dates:
            continue

        coupon_accrued = 0.0
        total_coupon_paid = 0.0
        called_at = None
        call_obs_num = None

        for i, obs in enumerate(obs_dates, start=1):
            s_obs = px.loc[obs, tickers].astype(float)
            worstof = float((s_obs / s0).min())

            # Coupon
            if worstof >= params.coupon_barrier:
                if params.memory:
                    total_coupon_paid += coupon_accrued + params.coupon
                    coupon_accrued = 0.0
                else:
                    total_coupon_paid += params.coupon
            else:
                if params.memory:
                    coupon_accrued += params.coupon

            # Autocall
            if worstof >= params.call_barrier:
                called_at = obs
                call_obs_num = i
                break

        if called_at is None:
            maturity = obs_dates[-1]
            s_m = px.loc[maturity, tickers].astype(float)
            worstof_m = float((s_m / s0).min())

            if worstof_m >= params.principal_barrier:
                principal = 1.0
            else:
                principal = 1.0 - params.leverage_down * (1.0 - worstof_m)
                principal = max(0.0, principal)

            end_dt = maturity
            payoff = principal + total_coupon_paid
            called = False
            obs_num = None
        else:
            end_dt = called_at
            payoff = 1.0 + total_coupon_paid
            called = True
            obs_num = call_obs_num

        life_m = months_between(launch, end_dt)
        if life_m <= 0:
            continue

        ann_linear = (payoff - 1.0) / (life_m / 12.0)

        rows.append({
            "launch": launch.date().isoformat(),
            "end": end_dt.date().isoformat(),
            "life_months": life_m,
            "payoff_total": payoff,
            "ann_linear": ann_linear,
            "called": called,
            "call_obs_num": obs_num,
            "positive": payoff >= 1.0
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Бэктест не дал траекторий. Проверь шаг запусков/период/тикеры.")
    return df

def summarize(df: pd.DataFrame) -> Dict[str, object]:
    avg_life = float(df["life_months"].mean())
    avg_ann = float(df["ann_linear"].mean())
    called_share = float(df["called"].mean())
    positive_share = float(df["positive"].mean())

    calls = df[df["called"] & df["call_obs_num"].notna()].copy()
    by_obs = calls["call_obs_num"].value_counts(normalize=True).sort_index().to_dict()

    return {
        "n_paths": int(len(df)),
        "avg_life_months": avg_life,
        "avg_ann_linear": avg_ann,
        "called_share": called_share,
        "positive_share": positive_share,
        "call_share_by_obs": by_obs,
    }


# =========================
# UI
# =========================

st.set_page_config(page_title="Backtester (Yahoo + Bloomberg tickers)", layout="wide")
st.title("Бэктест структурных продуктов — Yahoo Finance, ввод тикеров в формате Bloomberg")
st.caption("Вводите тикеры как в Bloomberg (например: BMW GY, NVDA US, 7203 JP, 0700 HK).")

colA, colB = st.columns([1.3, 1])
with colA:
    bbg_raw = st.text_area(
        "Базовые активы (Bloomberg tickers, по одному в строке)",
        value="NVDA US\nAAPL US\nBMW GY\n0700 HK\n7203 JP"
    )
with colB:
    start_s = st.text_input("Начало истории (YYYY-MM-DD)", value="2018-01-01")
    end_s = st.text_input("Конец истории (YYYY-MM-DD)", value="2025-12-31")
    launch_step = st.number_input("Запуск каждые N торговых дней", min_value=1, max_value=30, value=5)

st.markdown("### Конвертер тикеров Bloomberg → Yahoo")
overrides_text = st.text_area(
    "Overrides (если конвертер ошибся). Формат: `BBG = YAHOO`",
    value="",
    placeholder="Напр:\nBHP AU = BHP.AX\nSIE GY = SIE.DE\nBRK/B US = BRK-B"
)
overrides = parse_overrides(overrides_text)

# Normalize input tickers
bbg_list = [normalize_bbg_line(x) for x in bbg_raw.splitlines()]
bbg_list = [x for x in bbg_list if x]

if not bbg_list:
    st.stop()

# Convert
conv_rows = []
yahoo_symbols = []
for bbg in bbg_list:
    y, note = bbg_to_yahoo(bbg, overrides)
    yahoo_symbols.append(y)
    conv_rows.append({"Bloomberg": bbg, "Yahoo": y, "Note": note})

conv_df = pd.DataFrame(conv_rows)
st.dataframe(conv_df, use_container_width=True)

unknown = conv_df[conv_df["Note"].astype(str).str.startswith("unknown_exchange_code")]
if not unknown.empty:
    st.warning(
        "Есть тикеры с неизвестным кодом биржи (unknown_exchange_code). "
        "Добавьте overrides (BBG=YAHOO) для них — и дальше всё будет работать стабильно."
    )

st.subheader("Параметры продукта (MVP): Phoenix autocall worst-of")
c1, c2, c3, c4 = st.columns(4)
with c1:
    obs_months = st.selectbox("Частота наблюдений (мес)", [1, 3, 6], index=1)
    term_months = st.selectbox("Срок (мес)", [12, 18, 24, 36, 48], index=3)
with c2:
    call_barrier = st.number_input("Барьер отзыва (1.0 = 100%)", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
    principal_barrier = st.number_input("Барьер защиты (0.6 = 60%)", min_value=0.0, max_value=1.5, value=0.6, step=0.05)
with c3:
    coupon = st.number_input("Купон за период (0.03 = 3%)", min_value=0.0, max_value=0.3, value=0.03, step=0.005)
    coupon_barrier = st.number_input("Барьер купона (0.6 = 60%)", min_value=0.0, max_value=1.5, value=0.6, step=0.05)
with c4:
    memory = st.checkbox("Memory (накопление купонов)", value=True)
    leverage_down = st.number_input("Участие в падении ниже барьера (1x/2x)", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

run = st.button("Скачать котировки и запустить бэктест")

if run:
    try:
        start_dt = parse_date(start_s)
        end_dt = parse_date(end_s)
        if end_dt <= start_dt:
            st.error("Конец должен быть позже начала.")
            st.stop()

        # Basic validation: empty Yahoo symbol
        bad = [bbg for bbg, y in zip(bbg_list, yahoo_symbols) if not y]
        if bad:
            st.error(f"Не удалось получить Yahoo-символы для: {bad}")
            st.stop()

        params = PhoenixParams(
            obs_months=int(obs_months),
            term_months=int(term_months),
            call_barrier=float(call_barrier),
            coupon=float(coupon),
            coupon_barrier=float(coupon_barrier),
            memory=bool(memory),
            principal_barrier=float(principal_barrier),
            leverage_down=float(leverage_down),
        )

        with st.spinner("Скачиваю котировки с Yahoo Finance..."):
            px = download_prices(yahoo_symbols, start=str(start_dt), end=str(end_dt))

        # Rename columns back to Bloomberg tickers for readability
        rename_map = {y: b for b, y in zip(bbg_list, yahoo_symbols)}
        px = px.rename(columns=rename_map)

        with st.spinner("Считаю бэктест..."):
            df = backtest_phoenix_worstof(
                px=px,
                start_date=pd.Timestamp(start_dt),
                end_date=pd.Timestamp(end_dt),
                params=params,
                launch_step=int(launch_step),
            )
            summ = summarize(df)

        st.success(f"Готово. Траекторий: {summ['n_paths']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Средний срок жизни (мес)", f"{summ['avg_life_months']:.2f}")
        m2.metric("Средняя годовая доходность (линейная)", f"{summ['avg_ann_linear']*100:.2f}%")
        m3.metric("Доля отзывов", f"{summ['called_share']*100:.1f}%")
        m4.metric("Доля положительных исходов", f"{summ['positive_share']*100:.1f}%")

        st.subheader("Доля отзывов по наблюдениям")
        by_obs = summ["call_share_by_obs"]
        if by_obs:
            obs_df = pd.DataFrame({"Наблюдение": list(by_obs.keys()), "Доля отзывов": list(by_obs.values())})
            obs_df["Доля отзывов"] = (obs_df["Доля отзывов"] * 100).round(2).astype(str) + "%"
            st.dataframe(obs_df, use_container_width=True)
        else:
            st.info("Отзывов не было (в этом диапазоне/параметрах).")

        st.subheader("Пример котировок (первые 10 строк общих дат)")
        st.dataframe(px.dropna(how="any").head(10), use_container_width=True)

        st.subheader("Детализация траекторий (первые 200)")
        st.dataframe(df.head(200), use_container_width=True)

        st.download_button(
            "Скачать результаты бэктеста (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="backtest_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.info(
            "Типовые причины:\n"
            "- Некорректный Yahoo-символ (добавьте override)\n"
            "- Мало пересечения дат по тикерам\n"
            "- Слишком короткий диапазон или слишком большой шаг запусков\n"
        )
