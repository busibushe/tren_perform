# backup suc
import os
os.environ["WATCHFILES_DISABLE_INOTIFY"] = "1"

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats

# ==============================================================================
# KONFIGURASI HALAMAN & FUNGSI-FUNGSI
# ==============================================================================

st.set_page_config(
    page_title="Dashboard Performa Dinamis", 
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_sales_data(file):
    """
    Memuat dan membersihkan data penjualan dari file yang diunggah.
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        required_cols = {'Date': 'Sales Date', 'Branch': 'Branch', 'Menu': 'Menu', 'Bill Number': 'Bill Number'}
        for col, new_name in required_cols.items():
            if col in df.columns and new_name not in df.columns:
                 df.rename(columns={col: new_name}, inplace=True)
            if new_name not in df.columns:
                 raise ValueError(f"Kolom wajib '{new_name}' tidak ditemukan di file penjualan.")

        df['Sales Date'] = pd.to_datetime(df['Sales Date']).dt.date
        df['Branch'] = df['Branch'].fillna('Tidak Diketahui')

        numeric_cols = ['Qty', 'Price', 'Subtotal', 'Discount', 'Service Charge', 'Tax', 'VAT', 'Total', 'Nett Sales', 'Bill Discount', 'Total After Bill Discount']
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        
        df.fillna({col: 0 for col in numeric_cols if col in df.columns}, inplace=True)
        return df

    except Exception as e:
        raise ValueError(f"Terjadi kesalahan saat memproses file penjualan: {e}")

@st.cache_data
def load_additional_data(file):
    """
    Memuat data metrik tambahan yang dinamis dari file Excel.
    Secara otomatis mendeteksi kolom metrik (selain 'Date' dan 'Branch').
    """
    try:
        df_add = pd.read_excel(file)

        if 'Date' not in df_add.columns:
            raise ValueError("Kolom 'Date' tidak ditemukan di file metrik tambahan.")
        if 'Branch' not in df_add.columns:
            raise ValueError("Kolom 'Branch' tidak ditemukan di file metrik tambahan.")

        df_add['Date'] = pd.to_datetime(df_add['Date']).dt.date
        df_add['Branch'] = df_add['Branch'].fillna('Tidak Diketahui')

        metric_cols = [col for col in df_add.columns if col not in ['Date', 'Branch']]
        if not metric_cols:
            raise ValueError("Tidak ada kolom metrik yang terdeteksi di file. Pastikan ada kolom selain 'Date' dan 'Branch'.")

        for col in metric_cols:
            df_add[col] = pd.to_numeric(df_add[col], errors='coerce')
        
        df_add.fillna({col: 0 for col in metric_cols}, inplace=True)
        return df_add, metric_cols

    except Exception as e:
        raise ValueError(f"Terjadi kesalahan saat memproses file metrik tambahan: {e}")


def analyze_trend_v2(data_series, time_series):
    """
    Menganalisis tren dan mengembalikan narasi, garis tren, dan p-value untuk penjelasan.
    """
    if len(data_series.dropna()) < 3:
        return "Data tidak cukup untuk analisis tren (dibutuhkan minimal 3 bulan).", None, None
    
    if len(set(data_series.dropna())) <= 1:
        return "Data konstan, tidak ada tren yang bisa dianalisis.", None, None

    data_series_interpolated = data_series.interpolate(method='linear', limit_direction='both')
    
    x = np.arange(len(data_series_interpolated))
    y = data_series_interpolated.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trendline = slope * x + intercept

    if p_value < 0.05:
        overall_trend = f"menunjukkan tren **{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"
    else:
        overall_trend = "cenderung **stabil/fluktuatif** tanpa tren yang jelas"

    monthly_changes = data_series.pct_change().dropna()
    event_info = ""
    if not monthly_changes.empty and monthly_changes.std() > 0:
        significant_change_threshold = 1.5 * monthly_changes.std()
        max_increase_month_idx = monthly_changes.idxmax()
        if monthly_changes.max() > significant_change_threshold:
            event_info += f" Lonjakan tertinggi terjadi pada **{time_series[max_increase_month_idx]}**."
        max_decrease_month_idx = monthly_changes.idxmin()
        if abs(monthly_changes.min()) > significant_change_threshold:
            event_info += f" Penurunan tertajam terjadi pada **{time_series[max_decrease_month_idx]}**."
    
    narrative = f"Secara keseluruhan, data {overall_trend}.{event_info}"
    return narrative, trendline, p_value


def display_analysis_with_details(title, analysis_text, p_value):
    """Menampilkan analisis utama dan penjelasan p-value dalam expander."""
    st.info(f"💡 **{title}:** {analysis_text}")
    if p_value is not None:
        with st.expander("Lihat penjelasan p-value"):
            st.markdown(f"**Nilai p-value** tren ini adalah **`{p_value:.4f}`**. Angka ini berarti ada **`{p_value:.2%}`** kemungkinan melihat pola ini hanya karena kebetulan.")
            if p_value < 0.05:
                st.success("✔️ Karena kemungkinan kebetulan sangat rendah (< 5%), tren ini dianggap **nyata secara statistik**.")
            else:
                st.warning("⚠️ Karena kemungkinan kebetulan cukup tinggi (≥ 5%), tren ini **tidak signifikan secara statistik**.")
    st.markdown("---")

# ==============================================================================
# LOGIKA AUTENTIKASI
# ==============================================================================

config = {'credentials': st.secrets['credentials'].to_dict(), 'cookie': st.secrets['cookie'].to_dict()}
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)
name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("Username atau password salah.")
elif auth_status is None:
    st.warning("Silakan masukkan username dan password.")
    st.stop()
elif auth_status:
    # ==============================================================================
    # APLIKASI UTAMA (SETELAH LOGIN BERHASIL)
    # ==============================================================================

    # --- PERBAIKAN: Cara Stabil Menampilkan Logo ---
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    else:
        st.sidebar.warning("Logo 'logo.png' tidak ditemukan.")

    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{name}**")

    st.sidebar.title("📤 Unggah Data")
    uploaded_sales_file = st.sidebar.file_uploader("1. Unggah File Penjualan", type=["xlsx", "xls", "csv"])
    uploaded_metrics_file = st.sidebar.file_uploader("2. Unggah File Metrik Tambahan (Opsional)", type=["xlsx", "xls"])
    
    if uploaded_sales_file is None:
        st.info("Selamat datang! Silakan unggah file data penjualan Anda untuk memulai analisis.")
        st.stop()
    
    try:
        df = load_sales_data(uploaded_sales_file)
        df_add, metric_cols = (None, [])
        if uploaded_metrics_file:
            df_add, metric_cols = load_additional_data(uploaded_metrics_file)
    except ValueError as e:
        st.error(f"❌ **Error:** {e}")
        st.stop()

    # --- UI Filter ---
    st.sidebar.title("⚙️ Filter & Opsi")
    unique_branches = sorted(df['Branch'].unique())
    selected_branch = st.sidebar.selectbox("Pilih Cabang", unique_branches)
    min_date = df['Sales Date'].min()
    max_date = df['Sales Date'].max()
    date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    selected_metrics = []
    if metric_cols:
        selected_metrics = st.sidebar.multiselect("Pilih Metrik Tambahan", options=metric_cols, default=metric_cols)

    if len(date_range) != 2:
        st.warning("Mohon pilih rentang tanggal yang valid.")
        st.stop()
    
    start_date, end_date = date_range
    df_filtered = df[(df['Branch'] == selected_branch) & (df['Sales Date'] >= start_date) & (df['Sales Date'] <= end_date)]
    
    if df_filtered.empty:
        st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih.")
        st.stop()

    # --- Tampilan Dashboard ---
    st.title(f"📊 Dashboard Performa: {selected_branch}")
    st.markdown(f"Analisis data dari **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")
    st.markdown("---")

    # --- Proses Agregasi Data ---
    monthly_df = df_filtered.copy()
    monthly_df['Bulan'] = pd.to_datetime(monthly_df['Sales Date']).dt.to_period('M')
    monthly_agg = monthly_df.groupby('Bulan').agg(
        TotalMonthlySales=('Nett Sales', 'sum'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()
    monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)

    # Agregasi dan Penggabungan Data Tambahan jika ada
    if df_add is not None and selected_metrics:
        df_add_filtered = df_add[(df_add['Branch'] == selected_branch) & (df_add['Date'] >= start_date) & (df_add['Date'] <= end_date)]
        if not df_add_filtered.empty:
            add_monthly_df = df_add_filtered.copy()
            add_monthly_df['Bulan'] = pd.to_datetime(add_monthly_df['Date']).dt.to_period('M')
            
            agg_dict = {metric: 'mean' for metric in selected_metrics}
            additional_monthly_agg = add_monthly_df.groupby('Bulan').agg(agg_dict).reset_index()
            
            monthly_agg = pd.merge(monthly_agg, additional_monthly_agg, on='Bulan', how='left')

    if not monthly_agg.empty:
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()
    
    # --- Tampilan KPI Dinamis ---
    if not monthly_agg.empty:
        num_kpi_cols = 3 + len(selected_metrics)
        kpi_cols = st.columns(num_kpi_cols)

        last_month = monthly_agg.iloc[-1]
        prev_month = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None

        def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
            delta = (current_val - prev_val) / prev_val if prev_val and prev_val > 0 else 0
            val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.0f}"
            col.metric(title, val_format, f"{delta:.1%}" if prev_val else None, help=help_text if prev_val else None)
        
        help_str = f"Dibandingkan bulan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""
        display_kpi(kpi_cols[0], "💰 Penjualan", last_month['TotalMonthlySales'], prev_month['TotalMonthlySales'] if prev_month is not None else None, help_str, True)
        display_kpi(kpi_cols[1], "🛒 Transaksi", last_month['TotalTransactions'], prev_month['TotalTransactions'] if prev_month is not None else None, help_str, False)
        display_kpi(kpi_cols[2], "💳 AOV", last_month['AOV'], prev_month['AOV'] if prev_month is not None else None, help_str, True)
        
        for i, metric in enumerate(selected_metrics):
            if metric in last_month:
                display_kpi(kpi_cols[3+i], f"⭐ {metric}", last_month[metric], prev_month[metric] if prev_month is not None else None, help_str, False)
    
    st.markdown("---")

    # --- Tampilan Visualisasi Dinamis ---
    # Menambahkan tabel "Menu Terlaris" sebelum tab visualisasi lainnya
    with st.expander("📈 Lihat Menu Terlaris", expanded=False):
        top_menus = df_filtered.groupby('Menu')['Qty'].sum().sort_values(ascending=False).reset_index().head(10)
        top_menus.index = top_menus.index + 1
        st.dataframe(top_menus, use_container_width=True)
    
    tab_titles = ["Penjualan", "Transaksi", "AOV"] + selected_metrics
    tabs = st.tabs([f"**{title}**" for title in tab_titles])

    def create_trend_chart(tab, data, y_col, y_label, color):
        with tab:
            st.subheader(f"Analisis Tren Bulanan: {y_label}")
            fig = px.line(data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
            fig.update_traces(line_color=color)
            
            analysis, trendline, p_val = analyze_trend_v2(data[y_col], data['Bulan'].dt.strftime('%b %Y'))
            if trendline is not None:
                fig.add_scatter(x=data['Bulan'], y=trendline, mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
            
            st.plotly_chart(fig, use_container_width=True)
            display_analysis_with_details(f"Analisis Tren {y_label}", analysis, p_val)

    if not monthly_agg.empty:
        create_trend_chart(tabs[0], monthly_agg, 'TotalMonthlySales', 'Total Penjualan (Rp)', 'royalblue')
        create_trend_chart(tabs[1], monthly_agg, 'TotalTransactions', 'Jumlah Transaksi', 'orange')
        create_trend_chart(tabs[2], monthly_agg, 'AOV', 'Average Order Value (Rp)', 'green')

        for i, metric in enumerate(selected_metrics):
            if metric in monthly_agg.columns and monthly_agg[metric].notna().any():
                color_palette = px.colors.qualitative.Vivid
                color = color_palette[i % len(color_palette)]
                create_trend_chart(tabs[3+i], monthly_agg, metric, metric, color)