# backup tanpa komplain
import os
import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ==============================================================================
# KONFIGURASI APLIKASI
# ==============================================================================
st.set_page_config(
    page_title="Dashboard F&B Holistik",
    page_icon="🚀",
    layout="wide"
)

# Konfigurasi mapping kolom untuk mempermudah pengelolaan
REQUIRED_COLS_MAP = {
    'Sales Date': 'Tgl. Transaksi',
    'Branch': 'Nama Cabang',
    'Bill Number': 'No. Struk/Bill',
    'Nett Sales': 'Penjualan Bersih',
    'Menu': 'Nama Item/Menu',
    'Qty': 'Kuantitas'
}
OPTIONAL_COLS_MAP = {
    'Visit Purpose': 'Saluran Penjualan',
    'Payment Method': 'Metode Pembayaran',
    'Sales Date In': 'Waktu Pesanan Masuk',
    'Sales Date Out': 'Waktu Pesanan Selesai',
    'Order Time': 'Jam Pesanan',
    'Waiter': 'Nama Pramusaji/Waiter',
    'Discount': 'Diskon per Item',
    'Bill Discount': 'Diskon per Struk/Bill',
    'City': 'Kota'
}

# ==============================================================================
# FUNGSI PEMUATAN DATA
# ==============================================================================
@st.cache_data
def load_feather_file(uploaded_file):
    """Memuat satu file Feather yang sudah digabungkan."""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_feather(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Feather: {e}")
        return None

def find_best_column_match(all_columns, internal_name, description):
    """Mencari nama kolom yang paling cocok dari daftar berdasarkan nama internal dan deskripsi."""
    keywords = [
        internal_name.lower().replace("_", "").replace(" ", ""),
        description.lower().replace("_", "").replace(" ", "")
    ]
    keywords = list(set(keywords))
    for col in all_columns:
        if not col or not isinstance(col, str):
            continue
        cleaned_col = col.lower().replace("_", "").replace(" ", "")
        for keyword in keywords:
            if keyword in cleaned_col:
                return col
    return None

def process_mapped_data(df_raw, user_mapping):
    """
    Memproses DataFrame mentah setelah pemetaan kolom, dengan penanganan
    khusus untuk tipe data 'category'.
    """
    try:
        df = pd.DataFrame()
        # Salin kolom yang relevan dari data mentah
        for internal_name, source_col in user_mapping.items():
            if source_col and source_col in df_raw.columns:
                df[internal_name] = df_raw[source_col]

        # --- Konversi Tipe Data Numerik dan Tanggal (Tetap Sama) ---
        df['Sales Date'] = pd.to_datetime(df['Sales Date'], errors='coerce')
        
        numeric_cols = ['Qty', 'Nett Sales', 'Discount', 'Bill Discount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        for col in ['Sales Date In', 'Sales Date Out', 'Order Time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # --- PERBAIKAN UTAMA: Penanganan Kolom String & Kategori ---
        string_cols = ['Menu', 'Branch', 'Visit Purpose', 'Payment Method', 'Waiter', 'City']
        for col in string_cols:
            if col in df.columns:
                # Cek apakah kolom ini bertipe 'category' dari hasil optimisasi
                if pd.api.types.is_categorical_dtype(df[col]):
                    # Jika 'N/A' belum menjadi kategori yang valid, tambahkan.
                    if 'N/A' not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories(['N/A'])
                    
                    # Sekarang aman untuk mengisi nilai kosong dengan 'N/A'
                    df[col] = df[col].fillna('N/A')
                else:
                    # Jika bukan kategori, gunakan metode lama (ubah ke string)
                    df[col] = df[col].fillna('N/A').astype(str)

        # Hapus baris di mana tanggal transaksi tidak valid
        df.dropna(subset=['Sales Date'], inplace=True)
        return df
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return None

def reset_processing_state():
    """Callback untuk mereset state jika file diubah."""
    st.session_state.data_processed = False
    for key in list(st.session_state.keys()):
        if key.startswith("map_") or key == 'multiselect_selected':
            del st.session_state[key]
            
def create_all_inclusive_multiselect(df, column_name):
    """Membuat widget st.multiselect yang menyertakan opsi "(All)"."""
    ALL_OPTION_LABEL = "(All)"
    unique_options = sorted(df[column_name].unique())
    all_options_with_all = [ALL_OPTION_LABEL] + unique_options

    if 'multiselect_selected' not in st.session_state:
        st.session_state.multiselect_selected = all_options_with_all

    previous_selection = st.session_state.multiselect_selected
    st.session_state.multiselect_selected = st.sidebar.multiselect("Pilih Cabang:", options=all_options_with_all, default=st.session_state.multiselect_selected)
    current_selection = st.session_state.multiselect_selected

    if ALL_OPTION_LABEL in current_selection and ALL_OPTION_LABEL not in previous_selection:
        st.session_state.multiselect_selected = all_options_with_all
        st.rerun()
    elif ALL_OPTION_LABEL not in current_selection and ALL_OPTION_LABEL in previous_selection:
        st.session_state.multiselect_selected = []
        st.rerun()
    elif set(unique_options).issubset(set(current_selection)) and ALL_OPTION_LABEL not in current_selection:
        st.session_state.multiselect_selected = all_options_with_all
        st.rerun()
    elif not set(unique_options).issubset(set(current_selection)) and ALL_OPTION_LABEL in current_selection:
        st.session_state.multiselect_selected = [opt for opt in current_selection if opt != ALL_OPTION_LABEL]
        st.rerun()
        
    return [opt for opt in st.session_state.multiselect_selected if opt != ALL_OPTION_LABEL]

# ==============================================================================
# FUNGSI-FUNGSI ANALISIS & VISUALISASI
# ==============================================================================
def analyze_monthly_trends(df_filtered):
    """Menghitung agregasi data bulanan untuk metrik kunci (Penjualan, Transaksi, AOV)."""
    monthly_df = df_filtered.copy()
    monthly_df['Bulan'] = monthly_df['Sales Date'].dt.to_period('M')
    monthly_agg = monthly_df.groupby('Bulan').agg(TotalMonthlySales=('Nett Sales', 'sum'), TotalTransactions=('Bill Number', 'nunique')).reset_index()
    if not monthly_agg.empty:
        monthly_agg['AOV'] = monthly_agg.apply(lambda row: row['TotalMonthlySales'] / row['TotalTransactions'] if row['TotalTransactions'] > 0 else 0, axis=1)
        monthly_agg['Bulan'] = monthly_agg['Bulan'].dt.to_timestamp()
    return monthly_agg
def analyze_trend_v3(df_monthly, metric_col, metric_label):
    """Menganalisis tren dengan wawasan bisnis F&B: Tren Linear, YoY, dan Momentum."""
    if len(df_monthly.dropna(subset=[metric_col])) < 3:
        return {'narrative': "Data tidak cukup untuk analisis tren (dibutuhkan minimal 3 bulan)."}
    df = df_monthly.dropna(subset=[metric_col]).copy()
    if len(set(df[metric_col])) <= 1:
        return {'narrative': "Data konstan, tidak ada tren yang bisa dianalisis."}
    df['x_val'] = np.arange(len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['x_val'], df[metric_col])
    trendline = slope * df['x_val'] + intercept
    trend_type = "stabil/fluktuatif"
    if p_value < 0.05:
        trend_type = f"**{'meningkat' if slope > 0 else 'menurun'}** secara signifikan"
    yoy_narrative = ""
    if len(df) >= 13:
        last_val, yoy_val = df.iloc[-1][metric_col], df.iloc[-13][metric_col]
        if yoy_val > 0:
            yoy_change = (last_val - yoy_val) / yoy_val
            yoy_performance = f"**tumbuh {yoy_change:.1%}**" if yoy_change > 0 else f"**menurun {abs(yoy_change):.1%}**"
            yoy_narrative = f" Dibandingkan bulan yang sama tahun lalu, performa bulan terakhir {yoy_performance}."
    ma_line, momentum_narrative = None, ""
    if len(df) >= 4:
        ma_line = df[metric_col].rolling(window=3, min_periods=1).mean()
        momentum_narrative = " Momentum jangka pendek (3 bulan terakhir) terlihat **positif**." if ma_line.iloc[-1] > ma_line.iloc[-2] else " Momentum jangka pendek menunjukkan **perlambatan**."
    max_perf_month = df.loc[df[metric_col].idxmax()]
    min_perf_month = df.loc[df[metric_col].idxmin()]
    extrema_narrative = f" Performa tertinggi tercatat pada **{max_perf_month['Bulan'].strftime('%B %Y')}** dan terendah pada **{min_perf_month['Bulan'].strftime('%B %Y')}**."
    full_narrative = f"Secara keseluruhan, tren {metric_label} cenderung {trend_type}.{momentum_narrative}{yoy_narrative}{extrema_narrative}"
    return {'narrative': full_narrative, 'trendline': trendline, 'ma_line': ma_line, 'p_value': p_value}
def calculate_channel_analysis(df):
    """Menghitung data untuk analisis saluran penjualan."""
    if 'Visit Purpose' not in df.columns or df['Visit Purpose'].nunique() < 1:
        return None
    channel_sales = df.groupby('Visit Purpose')['Nett Sales'].sum().sort_values(ascending=False)
    agg_data = df.groupby('Visit Purpose').agg(TotalSales=('Nett Sales', 'sum'),TotalBills=('Bill Number', 'nunique'))
    agg_data['AOV'] = agg_data['TotalSales'] / agg_data['TotalBills']
    aov_by_channel = agg_data['AOV'].sort_values(ascending=False)
    return {'channel_sales': channel_sales, 'aov_by_channel': aov_by_channel}
def calculate_menu_engineering(df):
    """Menghitung data untuk analisis performa menu (menu engineering)."""
    if 'Menu' not in df.columns or df['Menu'].nunique() < 4:
        return None
    menu_perf = df.groupby('Menu').agg(Qty=('Qty', 'sum'), NettSales=('Nett Sales', 'sum')).reset_index()
    avg_qty = menu_perf['Qty'].mean()
    avg_sales = menu_perf['NettSales'].mean()
    stars = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] > avg_sales)]
    workhorses = menu_perf[(menu_perf['Qty'] > avg_qty) & (menu_perf['NettSales'] <= avg_sales)]
    return {'data': menu_perf, 'avg_qty': avg_qty, 'avg_sales': avg_sales, 'stars': stars, 'workhorses': workhorses}
def old_calculate_operational_efficiency(df):
    """Menghitung data untuk analisis efisiensi operasional."""
    required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time', 'Bill Number']
    if not all(col in df.columns for col in required_cols):
        return None
        
    df_eff = df.dropna(subset=['Sales Date In', 'Sales Date Out', 'Order Time']).copy()
    if df_eff.empty: return None

    df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
    df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
    if df_eff.empty: return None

    # --- PERBAIKAN 2: Ekstrak jam secara langsung dari kolom datetime ---
    # Ini lebih aman dan menghilangkan warning
    df_eff['Hour'] = df_eff['Order Time'].dt.hour
    
    agg_by_hour = df_eff.groupby('Hour').agg(
        AvgPrepTime=('Prep Time (Seconds)', 'mean'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()
    
    correlation, p_value = None, None
    if len(agg_by_hour) > 2:
        correlation, p_value = stats.spearmanr(agg_by_hour['TotalTransactions'], agg_by_hour['AvgPrepTime'])
        
    return {
        'data': df_eff,
        'agg_by_hour': agg_by_hour,
        'kpis': {
            'mean': df_eff['Prep Time (Seconds)'].mean(),
            'min': df_eff['Prep Time (Seconds)'].min(),
            'max': df_eff['Prep Time (Seconds)'].max()
        },
        'stats': {
            'correlation': correlation,
            'p_value': p_value
        }
    }
def calculate_operational_efficiency(df):
    """
    Menghitung data untuk analisis efisiensi operasional, dengan mengecualikan
    data di luar jam operasional (09:00 - 21:00).
    """
    required_cols = ['Sales Date In', 'Sales Date Out', 'Order Time', 'Bill Number']
    if not all(col in df.columns for col in required_cols):
        return None
        
    df_eff = df.dropna(subset=['Sales Date In', 'Sales Date Out', 'Order Time']).copy()
    if df_eff.empty: return None

    # --- PERUBAHAN DI SINI: FILTER DATA BERDASARKAN JAM OPERASIONAL ---
    # Hanya sertakan data dari jam 9 pagi (>= 9) hingga jam 9 malam (<= 21).
    df_eff = df_eff[df_eff['Order Time'].dt.hour.between(9, 21)]
    # ----------------------------------------------------------------

    df_eff['Prep Time (Seconds)'] = (df_eff['Sales Date Out'] - df_eff['Sales Date In']).dt.total_seconds()
    
    # Filter outlier waktu persiapan (misal: lebih dari 1 jam)
    df_eff = df_eff[df_eff['Prep Time (Seconds)'].between(0, 3600)]
    
    # Jika setelah difilter data menjadi kosong, hentikan fungsi.
    if df_eff.empty: 
        st.warning("Tidak ada data operasional yang valid ditemukan pada jam 09:00 - 21:00.")
        return None

    df_eff['Hour'] = df_eff['Order Time'].dt.hour
    
    agg_by_hour = df_eff.groupby('Hour').agg(
        AvgPrepTime=('Prep Time (Seconds)', 'mean'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()
    
    correlation, p_value = None, None
    if len(agg_by_hour) > 2:
        correlation, p_value = stats.spearmanr(agg_by_hour['TotalTransactions'], agg_by_hour['AvgPrepTime'])
        
    return {
        'data': df_eff,
        'agg_by_hour': agg_by_hour,
        'kpis': {
            'mean': df_eff['Prep Time (Seconds)'].mean(),
            'min': df_eff['Prep Time (Seconds)'].min(),
            'max': df_eff['Prep Time (Seconds)'].max()
        },
        'stats': {
            'correlation': correlation,
            'p_value': p_value
        }
    }
def generate_executive_summary(monthly_agg, channel_results, menu_results, ops_results):
    """Menciptakan ringkasan eksekutif otomatis berdasarkan hasil analisis yang sudah dihitung."""
    analyses = {'Penjualan': analyze_trend_v3(monthly_agg, 'TotalMonthlySales', 'Penjualan'),'Transaksi': analyze_trend_v3(monthly_agg, 'TotalTransactions', 'Transaksi'),'AOV': analyze_trend_v3(monthly_agg, 'AOV', 'AOV')}
    score_details = {}
    trend_health_score, yoy_score = 0, 0
    yoy_change_value = None
    for metric, analysis in analyses.items():
        trend_score, momentum_score = 0, 0
        narrative = analysis.get('narrative', '')
        if "meningkat** secara signifikan" in narrative: trend_score = 2
        elif "menurun** secara signifikan" in narrative: trend_score = -2
        if "Momentum jangka pendek (3 bulan terakhir) terlihat **positif**" in narrative: momentum_score = 1
        elif "Momentum jangka pendek menunjukkan **perlambatan**" in narrative: momentum_score = -1
        score_details[metric] = {'total': trend_score + momentum_score, 'tren_jangka_panjang': trend_score, 'momentum_jangka_pendek': momentum_score}
        trend_health_score += trend_score + momentum_score
    df = monthly_agg.dropna(subset=['TotalMonthlySales'])
    if len(df) >= 13:
        last_val, yoy_val = df.iloc[-1]['TotalMonthlySales'], df.iloc[-13]['TotalMonthlySales']
        if yoy_val > 0:
            yoy_change_value = (last_val - yoy_val) / yoy_val
            if yoy_change_value > 0.15: yoy_score = 2
            elif yoy_change_value > 0.05: yoy_score = 1
            elif yoy_change_value < -0.15: yoy_score = -2
            elif yoy_change_value < -0.05: yoy_score = -1
    score_details['YoY'] = {'total': yoy_score, 'tren_jangka_panjang': yoy_score, 'momentum_jangka_pendek': 0}
    health_score = trend_health_score + yoy_score
    health_status, health_color = "Perlu Perhatian", "orange"
    if health_score > 5: health_status, health_color = "Sangat Baik", "green"
    elif health_score >= 2: health_status, health_color = "Baik", "green"
    elif health_score <= -5: health_status, health_color = "Waspada", "red"
    health_context_narrative = ""
    sales_up = "meningkat** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    sales_down = "menurun** secara signifikan" in analyses['Penjualan'].get('narrative', '')
    trx_up = "meningkat** secara signifikan" in analyses['Transaksi'].get('narrative', '')
    aov_up = "meningkat** secara signifikan" in analyses['AOV'].get('narrative', '')
    aov_down = "menurun** secara signifikan" in analyses['AOV'].get('narrative', '')
    if sales_up and aov_up and not trx_up:
        health_context_narrative = "💡 **Insight Kunci:** Pertumbuhan didorong oleh **nilai belanja yang lebih tinggi (AOV naik)**, bukan dari penambahan jumlah transaksi."
    elif sales_up and trx_up and aov_down:
        health_context_narrative = "⚠️ **Perhatian:** Penjualan naik karena **volume transaksi yang tinggi**, namun AOV turun. Ini bisa jadi sinyal **terlalu banyak diskon** atau pergeseran ke produk yang lebih murah."
    elif sales_down and trx_up and aov_down:
        health_context_narrative = "🚨 **Waspada:** Jumlah transaksi mungkin naik, tapi **penurunan AOV yang tajam** menekan total penjualan secara signifikan. Analisis strategi harga dan promosi."
    recommendations = []
    if menu_results:
        if not menu_results['stars'].empty:
            top_star = menu_results['stars'].nlargest(1, 'NettSales')['Menu'].iloc[0]
            recommendations.append(f"🌟 **Prioritaskan Bintang:** Fokuskan promosi pada **'{top_star}'**.")
        if not menu_results['workhorses'].empty:
            top_workhorse = menu_results['workhorses'].nlargest(1, 'Qty')['Menu'].iloc[0]
            recommendations.append(f"🐴 **Optimalkan Profit:** Menu **'{top_workhorse}'** sangat laku, pertimbangkan menaikkan harga atau buat paket bundling.")
    if channel_results:
        highest_contrib_channel = channel_results['channel_sales'].idxmax()
        highest_aov_channel = channel_results['aov_by_channel'].idxmax()
        if highest_contrib_channel == highest_aov_channel:
             recommendations.append(f"🏆 **Maksimalkan Saluran Utama:** Saluran **'{highest_contrib_channel}'** adalah kontributor terbesar DAN memiliki AOV tertinggi. Prioritaskan segalanya di sini!")
        else:
             recommendations.append(f"💰 **Jaga Kontributor Terbesar:** Pertahankan performa saluran **'{highest_contrib_channel}'** yang menjadi penyumbang pendapatan utama Anda.")
             recommendations.append(f"📈 **Tingkatkan Frekuensi Saluran AOV Tinggi:** Pelanggan di **'{highest_aov_channel}'** belanja paling banyak per transaksi. Buat program loyalitas untuk mereka.")
    if ops_results and ops_results['stats']['p_value'] is not None:
        if ops_results['stats']['p_value'] < 0.05 and ops_results['stats']['correlation'] > 0.3:
            peak_hour = ops_results['agg_by_hour'].loc[ops_results['agg_by_hour']['TotalTransactions'].idxmax()]['Hour']
            recommendations.append(f"⏱️ **Atasi Kepadatan:** Layanan melambat saat ramai (terbukti statistik). Tambah sumber daya pada jam puncak sekitar pukul **{int(peak_hour)}:00**.")
    return {"health_status": health_status, "health_color": health_color, "yoy_change": yoy_change_value,"trend_narrative": analyses['Penjualan'].get('narrative', 'Gagal menganalisis tren penjualan.'),"health_context_narrative": health_context_narrative, "score_details": score_details,"recommendations": recommendations, "next_focus": "Pantau dampak eksekusi rekomendasi pada AOV, Transaksi, dan Kecepatan Layanan."}
def calculate_price_group_analysis(df):
    """
    Menganalisis performa berdasarkan kelompok harga yang ditemukan
    secara otomatis menggunakan K-Means clustering.
    """
    if 'Menu' not in df.columns or 'Nett Sales' not in df.columns or 'Qty' not in df.columns:
        return None

    # 1. Hitung harga rata-rata per item menu
    menu_prices = df.groupby('Menu').agg(
        TotalSales=('Nett Sales', 'sum'),
        TotalQty=('Qty', 'sum')
    ).reset_index()
    
    # Hindari pembagian dengan nol
    menu_prices = menu_prices[menu_prices['TotalQty'] > 0]
    menu_prices['AvgPrice'] = menu_prices['TotalSales'] / menu_prices['TotalQty']
    
    # Analisis memerlukan setidaknya 4 menu yang berbeda
    if len(menu_prices) < 4:
        return None

    # 2. Lakukan K-Means Clustering untuk menemukan 4 kelompok harga
    # Menggunakan K-Means untuk mengelompokkan harga ke dalam 4 cluster
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    menu_prices['PriceGroupLabel'] = kmeans.fit_predict(menu_prices[['AvgPrice']])

    # 3. Buat label yang deskriptif untuk setiap kelompok
    # Cari harga rata-rata dari setiap cluster untuk menentuk urutannya
    cluster_centers = menu_prices.groupby('PriceGroupLabel')['AvgPrice'].mean().sort_values().index
    # Buat pemetaan dari label cluster (e.g., 3) ke nama yang mudah dibaca (e.g., 'Kelompok 1 (Termurah)')
    label_mapping = {center: f"Kelompok {i+1}" for i, center in enumerate(cluster_centers)}
    menu_prices['PriceGroup'] = menu_prices['PriceGroupLabel'].map(label_mapping)
    
    # Tambahkan nama urutan deskriptif
    price_order = [" (Termurah)", " (Menengah)", " (Mahal)", " (Termahal)"]
    sorted_groups = menu_prices.groupby('PriceGroup')['AvgPrice'].mean().sort_values().index
    final_label_map = {group: group + price_order[i] for i, group in enumerate(sorted_groups)}
    menu_prices['PriceGroup'] = menu_prices['PriceGroup'].map(final_label_map)

    # 4. Gabungkan informasi kelompok harga kembali ke data transaksi utama
    df_with_groups = pd.merge(df, menu_prices[['Menu', 'PriceGroup']], on='Menu', how='left')

    # 5. Agregasi hasil berdasarkan kelompok harga
    group_performance = df_with_groups.groupby('PriceGroup').agg(
        TotalSales=('Nett Sales', 'sum'),
        TotalQty=('Qty', 'sum')
    ).reset_index()
    
    # Urutkan berdasarkan urutan kelompok harga
    group_performance['sort_order'] = group_performance['PriceGroup'].str.extract('(\d+)').astype(int)
    group_performance = group_performance.sort_values('sort_order').drop(columns='sort_order')

    return group_performance

def display_price_group_analysis(analysis_results):
    """
    Menampilkan visualisasi dan insight untuk analisis kelompok harga.
    """
    st.subheader("📊 Analisis Kelompok Harga")
    if analysis_results is None or analysis_results.empty:
        st.warning("Data tidak cukup untuk melakukan analisis kelompok harga (diperlukan minimal 4 menu berbeda).")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart untuk Total Penjualan
    fig.add_trace(
        go.Bar(x=analysis_results['PriceGroup'], y=analysis_results['TotalSales'], name='Total Penjualan', marker_color='royalblue'),
        secondary_y=False,
    )

    # Line chart untuk Total Kuantitas Terjual
    fig.add_trace(
        go.Scatter(x=analysis_results['PriceGroup'], y=analysis_results['TotalQty'], name='Total Kuantitas', mode='lines+markers', line=dict(color='darkorange')),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Kontribusi Penjualan vs. Kuantitas per Kelompok Harga",
        xaxis_title="Kelompok Harga",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Total Penjualan (Rp)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Total Kuantitas Terjual</b>", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Cara Membaca Grafik Ini:**
    - **Bilah Biru (Penjualan):** Menunjukkan seberapa besar pendapatan yang dihasilkan oleh setiap kelompok harga.
    - **Garis Oranye (Kuantitas):** Menunjukkan seberapa populer atau sering produk dalam kelompok harga tersebut terjual.
    - **Insight Kunci:** Cari ketidaksesuaian. Contoh: Kelompok harga dengan penjualan tinggi tapi kuantitas rendah menandakan produk *high-margin*. Sebaliknya, kuantitas tinggi tapi penjualan rendah menandakan produk *volume-driver* yang mungkin bisa dioptimalkan harganya.
    """)
def display_executive_summary(summary):
    """Menampilkan ringkasan eksekutif dengan layout baris tunggal yang jernih."""
    def format_score_with_arrows(score):
        if score >= 2: return "<span style='color:green; font-size:1.1em;'>↑↑</span>"
        if score == 1: return "<span style='color:green; font-size:1.1em;'>↑</span>"
        if score == 0: return "<span style='color:blue; font-size:1.1em;'>→</span>"
        if score == -1: return "<span style='color:red; font-size:1.1em;'>↓</span>"
        if score <= -2: return "<span style='color:red; font-size:1.1em;'>↓↓</span>"
        return ""
    st.subheader("Ringkasan Eksekutif")
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.5, 1.2, 2.3])
        with col1:
            st.markdown("Status Kesehatan Bisnis")
            st.markdown(f'<div style="background-color:{summary["health_color"]}; color:white; font-weight:bold; padding: 10px; border-radius: 7px; text-align:center;">{summary["health_status"].upper()}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("Performa YoY")
            yoy_change = summary.get('yoy_change')
            if yoy_change is not None:
                color = 'green' if yoy_change > 0 else 'red'
                arrow = '↑' if yoy_change > 0 else '↓'
                display_val = f"<h3 style='color:{color}; margin:0;'>{arrow} {abs(yoy_change):.1%}</h3>"
            else:
                display_val = "<h3 style='margin:0;'>N/A</h3>"
            st.markdown(display_val, unsafe_allow_html=True)
            st.caption("vs. Tahun Lalu")
        with col3:
            st.markdown("Analisis Skor Kesehatan")
            sub_cols = st.columns(4)
            metrics_to_display = ['Penjualan', 'Transaksi', 'AOV', 'YoY']
            for metric_name, sub_col in zip(metrics_to_display, sub_cols):
                with sub_col:
                    details = summary['score_details'].get(metric_name)
                    if details:
                        total_arrow = format_score_with_arrows(details['total'])
                        trend_arrow = format_score_with_arrows(details['tren_jangka_panjang'])
                        momentum_arrow = format_score_with_arrows(details['momentum_jangka_pendek'])
                        html = f'<div style="line-height: 1.2;"><strong>{metric_name}</strong><br><span style="font-size: 2em; font-weight: bold;">{total_arrow}</span>'
                        if metric_name != 'YoY':
                            html += f'<br><small>Tren: {trend_arrow}, Mom: {momentum_arrow}</small>'
                        html += '</div>'
                        st.markdown(html, unsafe_allow_html=True)
    with st.expander("🔍 Lihat Narasi Lengkap & Rekomendasi Aksi"):
        st.markdown("**Narasi Tren Utama (Penjualan):**")
        st.write(summary['trend_narrative'])
        if summary.get('health_context_narrative'): st.markdown(summary['health_context_narrative'])
        st.markdown("---")
        if summary['recommendations']:
            st.markdown("**Rekomendasi Aksi Teratas:**")
            for rec in summary['recommendations']: st.markdown(f"- {rec}")
        else:
            st.markdown("**Rekomendasi Aksi Teratas:** Tidak ada rekomendasi aksi prioritas spesifik untuk periode ini.")
        st.markdown("---")
        st.success(f"🎯 **Fokus Bulan Depan:** {summary['next_focus']}")
def display_monthly_kpis(monthly_agg):
    """Menampilkan KPI bulanan teratas dengan perbandingan bulan sebelumnya."""
    kpi_cols = st.columns(3)
    last_month = monthly_agg.iloc[-1]
    prev_month = monthly_agg.iloc[-2] if len(monthly_agg) >= 2 else None
    def display_kpi(col, title, current_val, prev_val, help_text, is_currency=True):
        delta = None
        if prev_val is not None and pd.notna(prev_val) and prev_val > 0:
            delta = (current_val - prev_val) / prev_val
        val_format = f"Rp {current_val:,.0f}" if is_currency else f"{current_val:,.0f}"
        col.metric(title, val_format, f"{delta:.1%}" if delta is not None else None, help=help_text if delta is not None else None)
    help_str = f"Dibandingkan {prev_month['Bulan'].strftime('%b %Y')}" if prev_month is not None else ""
    display_kpi(kpi_cols[0], "💰 Penjualan Bulanan", last_month.get('TotalMonthlySales'), prev_month.get('TotalMonthlySales') if prev_month is not None else None, help_str, True)
    display_kpi(kpi_cols[1], "🛒 Transaksi Bulanan", last_month.get('TotalTransactions'), prev_month.get('TotalTransactions') if prev_month is not None else None, help_str, False)
    display_kpi(kpi_cols[2], "💳 AOV Bulanan", last_month.get('AOV'), prev_month.get('AOV') if prev_month is not None else None, help_str, True)
    st.markdown("---")
def display_trend_chart_and_analysis(df_data, y_col, y_label, color):
    """Membuat grafik tren lengkap dengan garis tren, moving average, dan narasi analisis."""
    analysis_result = analyze_trend_v3(df_data, y_col, y_label)
    fig = px.line(df_data, x='Bulan', y=y_col, markers=True, labels={'Bulan': 'Bulan', y_col: y_label})
    fig.update_traces(line_color=color, name=y_label)
    if analysis_result.get('trendline') is not None:
        fig.add_scatter(x=df_data['Bulan'], y=analysis_result['trendline'], mode='lines', name='Garis Tren', line=dict(color='red', dash='dash'))
    if analysis_result.get('ma_line') is not None:
        fig.add_scatter(x=df_data['Bulan'], y=analysis_result['ma_line'], mode='lines', name='3-Month Moving Avg.', line=dict(color='orange', dash='dot'))
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"💡 **Analisis Tren {y_label}:** {analysis_result.get('narrative', 'Analisis tidak tersedia.')}")
    p_value = analysis_result.get('p_value')
    if p_value is not None:
        with st.expander("Lihat penjelasan signifikansi statistik (p-value)"):
            st.markdown(f"**Nilai p-value** tren ini adalah **`{p_value:.4f}`**. Angka ini berarti ada **`{p_value:.2%}`** kemungkinan melihat pola ini hanya karena kebetulan.")
            if p_value < 0.05:
                st.success("✔️ Karena kemungkinan kebetulan rendah (< 5%), tren ini dianggap **nyata secara statistik**.")
            else:
                st.warning("⚠️ Karena kemungkinan kebetulan cukup tinggi (≥ 5%), tren ini **tidak signifikan secara statistik**.")
    st.markdown("---")
def display_channel_analysis(channel_results):
    """Menampilkan visualisasi untuk analisis saluran penjualan."""
    st.subheader("📊 Analisis Saluran Penjualan")
    if not channel_results:
        st.warning("Kolom 'Visit Purpose' tidak dipetakan atau tidak ada data. Analisis saluran penjualan tidak tersedia.")
        return
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(channel_results['channel_sales'], values='Nett Sales', names=channel_results['channel_sales'].index, title="Kontribusi Penjualan per Saluran", hole=0.4)
        fig.update_layout(legend_title_text='Saluran')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        aov_by_channel = channel_results['aov_by_channel']
        fig2 = px.bar(aov_by_channel, x=aov_by_channel.index, y=aov_by_channel.values, labels={'y': 'Rata-rata Nilai Pesanan (AOV)', 'x': 'Saluran'}, title="AOV per Saluran Penjualan")
        st.plotly_chart(fig2, use_container_width=True)
    highest_aov_channel = channel_results['aov_by_channel'].index[0]
    highest_contrib_channel = channel_results['channel_sales'].index[0]
    st.info(f"""
    **Insight Bisnis:**
    - **Kontributor Terbesar:** Saluran **{highest_contrib_channel}** adalah penyumbang pendapatan terbesar.
    - **Nilai Pesanan Tertinggi:** Pelanggan dari saluran **{highest_aov_channel}** cenderung menghabiskan lebih banyak per transaksi.
    """)
def display_menu_engineering(menu_results):
    """Menampilkan visualisasi kuadran untuk menu engineering."""
    st.subheader("🔬 Analisis Performa Menu")
    if not menu_results:
        st.warning("Data menu tidak cukup untuk analisis kuadran (dibutuhkan min. 4 menu)."); return
    menu_perf, avg_qty, avg_sales = menu_results['data'], menu_results['avg_qty'], menu_results['avg_sales']
    show_text = len(menu_perf) < 75
    fig = px.scatter(menu_perf, x='Qty', y='NettSales', text=menu_perf['Menu'] if show_text else None, title="Kuadran Performa Menu",
                     labels={'Qty': 'Total Kuantitas Terjual', 'NettSales': 'Total Penjualan Bersih (Rp)'},
                     size='NettSales', color='NettSales', hover_name='Menu')
    fig.add_shape(type="rect", x0=avg_qty, y0=avg_sales, x1=menu_perf['Qty'].max(), y1=menu_perf['NettSales'].max(), fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=menu_perf['Qty'].min(), y0=avg_sales, x1=avg_qty, y1=menu_perf['NettSales'].max(), fillcolor="lightblue", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=avg_qty, y0=menu_perf['NettSales'].min(), x1=menu_perf['Qty'].max(), y1=avg_sales, fillcolor="lightyellow", opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=menu_perf['Qty'].min(), y0=menu_perf['NettSales'].min(), x1=avg_qty, y1=avg_sales, fillcolor="lightcoral", opacity=0.2, layer="below", line_width=0)
    fig.add_vline(x=avg_qty, line_dash="dash", line_color="gray")
    fig.add_hline(y=avg_sales, line_dash="dash", line_color="gray")
    if show_text:
        fig.update_traces(textposition='top center')
    else:
        st.warning("⚠️ Label nama menu disembunyikan karena terlalu banyak. Arahkan mouse ke titik untuk melihat detail.")
    st.plotly_chart(fig, use_container_width=True)
    stars, workhorses = menu_results['stars'], menu_results['workhorses']
    insight_text = "**Rekomendasi Aksi:**\n"
    if not stars.empty:
        top_star = stars.nlargest(1, 'NettSales')['Menu'].iloc[0]
        insight_text += f"- **Fokuskan Promosi:** Menu **'{top_star}'** adalah Bintang (Star) utama Anda.\n"
    if not workhorses.empty:
        top_workhorse = workhorses.nlargest(1, 'Qty')['Menu'].iloc[0]
        insight_text += f"- **Peluang Profit:** Menu **'{top_workhorse}'** sangat populer (Workhorse). Coba naikkan harga atau buat paket bundling."
    if len(insight_text) > 30: st.info(insight_text)
    with st.expander("💡 Cara Membaca Kuadran Performa Menu"):
        st.markdown("""
        Grafik ini membantu Anda mengkategorikan item menu berdasarkan popularitas (sumbu X) dan profitabilitas (sumbu Y).
        - **<span style='color:green; font-weight:bold;'>STARS 🌟 (Kanan Atas):</span>** Juara Anda! Populer dan menguntungkan. **Aksi: Pertahankan!**
        - **<span style='color:darkgoldenrod; font-weight:bold;'>WORKHORSES 🐴 (Kanan Bawah):</span>** Populer, kurang profit. **Aksi: Naikkan harga atau bundling.**
        - **<span style='color:blue; font-weight:bold;'>PUZZLES 🤔 (Kiri Atas):</span>** Sangat profit, jarang dipesan. **Aksi: Promosikan atau latih staf.**
        - **<span style='color:red; font-weight:bold;'>DOGS 🐶 (Kiri Bawah):</span>** Kurang populer & profit. **Aksi: Pertimbangkan hapus dari menu.**
        """, unsafe_allow_html=True)
def display_operational_efficiency(ops_results):
    """Menampilkan visualisasi dan insight untuk efisiensi operasional."""
    st.subheader("⏱️ Analisis Efisiensi Operasional")
    if not ops_results:
        st.warning("Kolom waktu (In, Out, Order Time) atau Bill Number tidak dipetakan/tidak valid."); return
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Waktu Persiapan Rata-rata", f"{ops_results['kpis']['mean']:.1f} dtk")
    kpi_col2.metric("Waktu Persiapan Tercepat", f"{ops_results['kpis']['min']:.1f} dtk")
    kpi_col3.metric("Waktu Persiapan Terlama", f"{ops_results['kpis']['max']:.1f} dtk")
    agg_by_hour = ops_results['agg_by_hour']
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg_by_hour['Hour'], y=agg_by_hour['TotalTransactions'], name="Jumlah Transaksi"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg_by_hour['Hour'], y=agg_by_hour['AvgPrepTime'], name="Rata-rata Waktu Persiapan", mode='lines+markers'), secondary_y=True)
    fig.update_layout(title_text="Korelasi Waktu Persiapan & Jumlah Pengunjung per Jam")
    fig.update_xaxes(title_text="Jam dalam Sehari")
    fig.update_yaxes(title_text="<b>Jumlah Transaksi</b> (Batang)", secondary_y=False)
    fig.update_yaxes(title_text="<b>Rata-rata Waktu Persiapan (Detik)</b> (Garis)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    peak_visitor_hour = agg_by_hour.loc[agg_by_hour['TotalTransactions'].idxmax()]
    longest_prep_hour = agg_by_hour.loc[agg_by_hour['AvgPrepTime'].idxmax()]
    st.info(f"""
    **Insight Bisnis:**
    - **Jam Puncak Pengunjung:** Kepadatan tertinggi pada jam **{int(peak_visitor_hour['Hour'])}:00**.
    - **Layanan Melambat:** Waktu persiapan terlama pada jam **{int(longest_prep_hour['Hour'])}:00**.
    **Rekomendasi Aksi:** Jika jam layanan melambat **berdekatan** dengan jam puncak, pertimbangkan menambah staf atau menyederhanakan menu pada jam tersebut.
    """)
    with st.expander("🔬 Lihat Uji Statistik Korelasi"):
        corr = ops_results['stats']['correlation']
        p_val = ops_results['stats']['p_value']
        if corr is not None:
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Koefisien Korelasi Spearman (ρ)", f"{corr:.3f}")
            stat_col2.metric("P-value", f"{p_val:.3f}")
            strength = "lemah"
            if abs(corr) >= 0.7: strength = "sangat kuat"
            elif abs(corr) >= 0.5: strength = "kuat"
            elif abs(corr) >= 0.3: strength = "moderat"
            if p_val < 0.05:
                st.success(f"**Kesimpulan Statistik:** Terdapat korelasi positif yang **{strength} dan signifikan**. Ini membuktikan secara angka bahwa saat pengunjung ramai, layanan dapur cenderung melambat.")
            else:
                st.warning(f"**Kesimpulan Statistik:** Hasil ini **tidak signifikan secara statistik**. Kita tidak bisa menyimpulkan dengan yakin bahwa kepadatan pengunjung adalah penyebab layanan melambat berdasarkan data ini saja.")
        else:
            st.warning("Tidak cukup data per jam untuk melakukan uji korelasi statistik.")

def create_waiter_performance_analysis(df):
    """Menganalisis performa pramusaji melampaui total penjualan."""
    st.subheader("👨‍🍳 Analisis Kinerja Pramusaji (Waiter)")
    if 'Waiter' not in df.columns:
        st.warning("Kolom 'Waiter' tidak dipetakan. Analisis ini tidak tersedia.")
        return

    waiter_perf = df.groupby('Waiter').agg(
        TotalSales=('Nett Sales', 'sum'),
        TotalTransactions=('Bill Number', 'nunique'),
        TotalDiscount=('Discount', 'sum')
    ).reset_index()
    
    waiter_perf = waiter_perf[waiter_perf['TotalTransactions'] > 0].copy()
    if waiter_perf.empty:
        st.info("Tidak ada data kinerja pramusaji yang valid untuk ditampilkan.")
        return

    waiter_perf['AOV'] = waiter_perf['TotalSales'] / waiter_perf['TotalTransactions']
    waiter_perf['DiscountRate'] = (waiter_perf['TotalDiscount'] / (waiter_perf['TotalSales'] + waiter_perf['TotalDiscount'])).fillna(0)

    col1, col2, col3 = st.columns(3)
    top_sales = waiter_perf.nlargest(1, 'TotalSales')
    if not top_sales.empty:
        # --- PERBAIKAN 1 ---
        sales_value = float(top_sales['TotalSales'].iloc[0])
        col1.metric("Penjualan Tertinggi", top_sales['Waiter'].iloc[0], f"Rp {sales_value:,.0f}")

    top_aov = waiter_perf.nlargest(1, 'AOV')
    if not top_aov.empty:
        # --- PERBAIKAN 2 ---
        aov_value = float(top_aov['AOV'].iloc[0])
        col2.metric("AOV Tertinggi (Upseller)", top_aov['Waiter'].iloc[0], f"Rp {aov_value:,.0f}")
    
    discounter_perf = waiter_perf[waiter_perf['DiscountRate'] > 0]
    top_discounter = discounter_perf.nlargest(1, 'DiscountRate')
    if not top_discounter.empty:
        # --- PERBAIKAN 3 (Paling Kritis) ---
        discount_rate_value = float(top_discounter['DiscountRate'].iloc[0])
        col3.metric("Rasio Diskon Tertinggi", top_discounter['Waiter'].iloc[0], f"{discount_rate_value:.1%}")
    else:
        col3.metric("Rasio Diskon Tertinggi", "-", "Tidak Ada Diskon")


    st.info("""
    **Insight Aksi:**
    - **Pahlawan AOV vs. Penjualan**: Perhatikan pramusaji dengan AOV tertinggi, bukan hanya penjualan tertinggi. Mereka adalah *upseller* terbaik Anda. Jadikan mereka contoh atau mentor.
    - **Waspadai 'Raja Diskon'**: Selidiki mengapa pramusaji dengan rasio diskon tertinggi sering memberikan diskon. Apakah itu strategi atau masalah yang perlu ditangani?
    """)
    with st.expander("Lihat Data Kinerja Pramusaji Lengkap"):
        st.dataframe(waiter_perf.sort_values('TotalSales', ascending=False), use_container_width=True)
def create_discount_effectiveness_analysis(df):
    """
    Menganalisis apakah diskon efektif meningkatkan belanja pelanggan.
    Fungsi ini membandingkan Average Order Value (AOV) antara transaksi
    yang menggunakan diskon dengan yang tidak.
    """
    st.subheader("📉 Analisis Efektivitas Diskon")

    # Memastikan kolom yang dibutuhkan tersedia di data
    if 'Discount' not in df.columns or 'Bill Discount' not in df.columns:
        st.warning("Kolom 'Discount' dan/atau 'Bill Discount' tidak dipetakan. Analisis ini tidak tersedia.")
        return
        
    df_analysis = df.copy()
    
    # Menghitung total diskon per baris dan menandai transaksi mana yang punya diskon
    df_analysis['TotalDiscount'] = df_analysis['Discount'] + df_analysis['Bill Discount']
    df_analysis['HasDiscount'] = df_analysis['TotalDiscount'] > 0
    
    # Jika tidak ada variasi (semua transaksi punya diskon atau semua tidak punya), analisis tidak bisa dilanjutkan
    if df_analysis['HasDiscount'].nunique() < 2:
        st.info("Tidak ada cukup data untuk membandingkan transaksi dengan dan tanpa diskon.")
        return

    # Agregasi per nomor struk/bill untuk menghitung AOV dengan benar
    bill_agg = df_analysis.groupby('Bill Number').agg(
        NettSales=('Nett Sales', 'sum'),
        HasDiscount=('HasDiscount', 'max') # Jika satu item saja ada diskon, maka seluruh struk dianggap punya diskon
    )
    
    # Menghitung AOV rata-rata untuk grup 'dengan diskon' dan 'tanpa diskon'
    aov_comparison = bill_agg.groupby('HasDiscount')['NettSales'].mean()
    
    # Memeriksa apakah kedua grup ada untuk dibandingkan
    if True in aov_comparison.index and False in aov_comparison.index:
        
        # --- PERBAIKAN UTAMA ---
        # Mengubah tipe data dari numpy.float64 ke float standar Python untuk menghindari error format
        aov_with_discount = float(aov_comparison.loc[True])
        aov_without_discount = float(aov_comparison.loc[False])
        
        # Menampilkan perbandingan menggunakan st.metric
        col1, col2 = st.columns(2)
        col1.metric("AOV dengan Diskon", f"Rp {aov_with_discount:,.0f}")
        col2.metric("AOV tanpa Diskon", f"Rp {aov_without_discount:,.0f}")
        
        # Memberikan insight otomatis berdasarkan hasil perbandingan
        diff = (aov_with_discount - aov_without_discount) / aov_without_discount
        if diff > 0.05: # Jika AOV naik lebih dari 5%, dianggap efektif
            st.success(f"✅ **Efektif**: Pemberian diskon secara umum berhasil meningkatkan nilai belanja rata-rata sebesar **{diff:.1%}**.")
        else:
            st.warning(f"⚠️ **Potensi Kanibalisasi**: Diskon tidak meningkatkan AOV secara signifikan (perubahan {diff:.1%}). Ada risiko diskon hanya mengurangi profit tanpa mendorong pelanggan untuk belanja lebih banyak.")
    else:
        st.info("Tidak ada cukup data untuk membandingkan AOV dengan dan tanpa diskon.")
def create_regional_analysis(df):
    """Menganalisis perbedaan performa dan preferensi antar kota."""
    st.subheader("🏙️ Analisis Kinerja Regional")
    if 'City' not in df.columns or df['City'].nunique() < 2:
        st.warning("Kolom 'City' tidak dipetakan atau hanya ada satu kota. Analisis regional tidak tersedia.")
        return
        
    city_perf = df.groupby('City').agg(
        TotalSales=('Nett Sales', 'sum'),
        TotalTransactions=('Bill Number', 'nunique')
    ).reset_index()
    city_perf = city_perf[city_perf['TotalTransactions'] > 0]
    city_perf['AOV'] = city_perf['TotalSales'] / city_perf['TotalTransactions']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=city_perf['City'], y=city_perf['TotalSales'], name='Total Penjualan'), secondary_y=False)
    fig.add_trace(go.Scatter(x=city_perf['City'], y=city_perf['AOV'], name='AOV', mode='lines+markers'), secondary_y=True)
    
    fig.update_layout(
        title_text="Perbandingan Penjualan dan AOV antar Kota",
        xaxis_title="Kota",
        yaxis_title="<b>Total Penjualan (Rp)</b>",
        yaxis2_title="<b>AOV (Rp)</b>",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analisis preferensi menu per kota
    top_cities = city_perf.nlargest(min(3, len(city_perf)), 'TotalSales')['City'].tolist()
    if top_cities:
        st.markdown("**Preferensi Menu Teratas per Kota (Berdasarkan Kuantitas)**")
        cols = st.columns(len(top_cities))
        for i, city in enumerate(top_cities):
            with cols[i]:
                st.markdown(f"**📍 {city}**")
                top_menus = df[df['City'] == city].groupby('Menu')['Qty'].sum().nlargest(5).index.tolist()
                if not top_menus:
                    st.caption("Tidak ada data menu.")
                for menu in top_menus:
                    st.caption(f"- {menu}")

# ==============================================================================
# APLIKASI UTAMA STREAMLIT
# ==============================================================================
def old_main_app(user_name):
    # ... (kode otentikasi seperti biasa) ...
    st.sidebar.title("📤 Unggah & Mapping Kolom")

    # --- UBAH BAGIAN INI: HANYA UPLOAD SATU FILE ---
    uploaded_file = st.sidebar.file_uploader(
        "1. Unggah Database Lengkap (.feather)", 
        type=["feather"],
        accept_multiple_files=False, # Diubah menjadi False
        on_change=reset_processing_state
    )
    
    if uploaded_file is None:
        st.info("👋 Selamat datang! Silakan unggah file 'database_lengkap.feather' Anda.")
        st.stop()
        
    # Gunakan fungsi pemuatan yang baru dan sederhana
    df_raw = load_feather_file(uploaded_file)
    if df_raw is None: 
        st.stop()

    user_mapping = {}
    all_cols = [""] + df_raw.columns.tolist()
    with st.sidebar.expander("Atur Kolom Wajib", expanded=not st.session_state.data_processed):
        for internal_name, desc in REQUIRED_COLS_MAP.items():
            best_guess = find_best_column_match(all_cols, internal_name, desc)
            index = all_cols.index(best_guess) if best_guess else 0
            key = f"map_req_{internal_name}"
            user_mapping[internal_name] = st.selectbox(f"**{desc}**:", options=all_cols, index=index, key=key)
    with st.sidebar.expander("Atur Kolom Opsional"):
        for internal_name, desc in OPTIONAL_COLS_MAP.items():
            best_guess = find_best_column_match(all_cols, internal_name, desc)
            index = all_cols.index(best_guess) if best_guess else 0
            key = f"map_opt_{internal_name}"
            user_mapping[internal_name] = st.selectbox(f"**{desc}**:", options=all_cols, index=index, key=key)

    if st.sidebar.button("✅ Terapkan dan Proses Data", type="primary"):
        mapped_req_cols = [user_mapping.get(k) for k in REQUIRED_COLS_MAP.keys()]
        if not all(mapped_req_cols):
            st.error("❌ Harap petakan semua kolom WAJIB diisi."); st.stop()
        chosen_cols = [c for c in user_mapping.values() if c]
        if len(chosen_cols) != len(set(chosen_cols)):
            st.error("❌ Terdeteksi satu kolom dipilih untuk beberapa peran berbeda."); st.stop()
        
        with st.spinner("Memproses data... Ini mungkin memakan waktu beberapa saat."):
            df_processed = process_mapped_data(df_raw, user_mapping)
            if df_processed is not None:
                st.session_state.df_processed = df_processed
                st.session_state.data_processed = True
                if 'multiselect_selected' in st.session_state:
                    del st.session_state.multiselect_selected
                st.rerun()

    if st.session_state.data_processed:
        df_processed = st.session_state.df_processed
        
        # --- SIDEBAR: Filter Global ---
        st.sidebar.title("⚙️ Filter Global")
        
        # Menggunakan filter selectbox tunggal sementara
        ALL_BRANCHES_OPTION = "Semua Cabang (Gabungan)"
        unique_branches = sorted(df_processed['Branch'].unique())
        branch_options = [ALL_BRANCHES_OPTION] + unique_branches
        selected_branch = st.sidebar.selectbox("Pilih Cabang", branch_options)
        
        min_date, max_date = df_processed['Sales Date'].min().date(), df_processed['Sales Date'].max().date()
        date_range = st.sidebar.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

        if len(date_range) != 2: st.stop()
        
        start_date, end_date = date_range
        df_filtered_by_date = df_processed[(df_processed['Sales Date'].dt.date >= start_date) & (df_processed['Sales Date'].dt.date <= end_date)]
        if selected_branch == ALL_BRANCHES_OPTION:
            df_filtered = df_filtered_by_date
        else:
            df_filtered = df_filtered_by_date[df_filtered_by_date['Branch'] == selected_branch]

        if df_filtered.empty:
            st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih."); st.stop()
        
        st.title(f"Dashboard Analisis Penjualan: {selected_branch}")
        st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

        # --- LAKUKAN SEMUA ANALISIS SEKALI SAJA ---
        monthly_agg = analyze_monthly_trends(df_filtered)
        channel_results = calculate_channel_analysis(df_filtered)
        menu_results = calculate_menu_engineering(df_filtered)
        ops_results = calculate_operational_efficiency(df_filtered)
        
        # --- TAMPILKAN RINGKASAN EKSEKUTIF ---
        if monthly_agg is not None and len(monthly_agg) >= 3:
            summary = generate_executive_summary(monthly_agg, channel_results, menu_results, ops_results)
            display_executive_summary(summary)

        # --- Menambahkan tab ketiga "Analisis Strategis" ---
        trend_tab, ops_tab, strategic_tab = st.tabs([
            "📈 **Dashboard Tren Performa**", 
            "🚀 **Dashboard Analisis Operasional**",
            "🧠 **Analisis Strategis**"
        ])

        with trend_tab:
            st.header("Analisis Tren Performa Jangka Panjang")
            if monthly_agg is not None and not monthly_agg.empty:
                display_monthly_kpis(monthly_agg)
                display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
                display_trend_chart_and_analysis(monthly_agg, 'TotalTransactions', 'Transaksi', 'orange')
                display_trend_chart_and_analysis(monthly_agg, 'AOV', 'AOV', 'green')
            else:
                st.warning("Tidak ada data bulanan yang cukup untuk analisis tren pada periode ini.")

        with ops_tab:
            st.header("Wawasan Operasional dan Taktis")
            display_channel_analysis(channel_results)
            st.markdown("---")
            display_menu_engineering(menu_results)
            st.markdown("---")
            display_operational_efficiency(ops_results)

        # --- Mengisi tab baru dengan fungsi analisis strategis ---
        with strategic_tab:
            st.header("Sintesis dan Analisis Silang")
            create_waiter_performance_analysis(df_filtered.copy())
            st.markdown("---")
            create_discount_effectiveness_analysis(df_filtered.copy())
            st.markdown("---")
            create_regional_analysis(df_filtered.copy())
def main_app(user_name):
    """
    Fungsi utama yang menjalankan seluruh aplikasi dashboard,
    dari unggah file hingga menampilkan semua analisis.
    """
    # Inisialisasi session state untuk status pemrosesan data
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False

    # ==========================================================================
    # SIDEBAR: Autentikasi dan Unggah File
    # ==========================================================================
    if os.path.exists("logo.png"): 
        st.sidebar.image("logo.png", width=150)
    
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Login sebagai: **{user_name}**")
    st.sidebar.title("📤 Unggah & Mapping Kolom")

    # Widget untuk mengunggah satu file Feather yang sudah diproses
    uploaded_file = st.sidebar.file_uploader(
        "1. Unggah Database Lengkap (.feather)", 
        type=["feather"],
        accept_multiple_files=False,
        on_change=reset_processing_state
    )
    
    if uploaded_file is None:
        st.info("👋 Selamat datang! Silakan unggah file 'database_lengkap.feather' Anda.")
        st.stop()
        
    # Memuat data dari file Feather
    df_raw = load_feather_file(uploaded_file)
    if df_raw is None: 
        st.stop()

    # ==========================================================================
    # SIDEBAR: Pemetaan Kolom
    # ==========================================================================
    user_mapping = {}
    all_cols = [""] + df_raw.columns.tolist()
    
    with st.sidebar.expander("Atur Kolom Wajib", expanded=not st.session_state.data_processed):
        for internal_name, desc in REQUIRED_COLS_MAP.items():
            best_guess = find_best_column_match(all_cols, internal_name, desc)
            index = all_cols.index(best_guess) if best_guess else 0
            key = f"map_req_{internal_name}"
            user_mapping[internal_name] = st.selectbox(f"**{desc}**:", options=all_cols, index=index, key=key)
            
    with st.sidebar.expander("Atur Kolom Opsional"):
        for internal_name, desc in OPTIONAL_COLS_MAP.items():
            best_guess = find_best_column_match(all_cols, internal_name, desc)
            index = all_cols.index(best_guess) if best_guess else 0
            key = f"map_opt_{internal_name}"
            user_mapping[internal_name] = st.selectbox(f"**{desc}**:", options=all_cols, index=index, key=key)

    # Tombol untuk memproses data setelah pemetaan
    if st.sidebar.button("✅ Terapkan dan Proses Data", type="primary"):
        mapped_req_cols = [user_mapping.get(k) for k in REQUIRED_COLS_MAP.keys()]
        if not all(mapped_req_cols):
            st.error("❌ Harap petakan semua kolom WAJIB diisi."); st.stop()
        
        chosen_cols = [c for c in user_mapping.values() if c]
        if len(chosen_cols) != len(set(chosen_cols)):
            st.error("❌ Terdeteksi satu kolom dipilih untuk beberapa peran berbeda."); st.stop()
        
        with st.spinner("Memproses data..."):
            df_processed = process_mapped_data(df_raw, user_mapping)
            if df_processed is not None:
                st.session_state.df_processed = df_processed
                st.session_state.data_processed = True
                if 'multiselect_selected' in st.session_state:
                    del st.session_state.multiselect_selected
                st.rerun()

    # ==========================================================================
    # KONTEN UTAMA: Tampil setelah data berhasil diproses
    # ==========================================================================
    if st.session_state.data_processed:
        df_processed = st.session_state.df_processed
        
        # --- SIDEBAR: Filter Global ---
        st.sidebar.title("⚙️ Filter Global")
        
        ALL_BRANCHES_OPTION = "Semua Cabang (Gabungan)"
        unique_branches = sorted(df_processed['Branch'].unique())
        branch_options = [ALL_BRANCHES_OPTION] + unique_branches
        selected_branch = st.sidebar.selectbox("Pilih Cabang", branch_options)
        
        min_date = df_processed['Sales Date'].min().date()
        max_date = df_processed['Sales Date'].max().date()
        date_range = st.sidebar.date_input(
            "Pilih Rentang Tanggal", 
            value=(min_date, max_date), 
            min_value=min_date, 
            max_value=max_date
        )

        if len(date_range) != 2: 
            st.stop()
        
        start_date, end_date = date_range
        
        # Menerapkan filter ke data yang telah diproses
        mask_date = (df_processed['Sales Date'].dt.date >= start_date) & \
                    (df_processed['Sales Date'].dt.date <= end_date)
        df_filtered_by_date = df_processed[mask_date]
        
        if selected_branch == ALL_BRANCHES_OPTION:
            df_filtered = df_filtered_by_date
        else:
            df_filtered = df_filtered_by_date[df_filtered_by_date['Branch'] == selected_branch]

        if df_filtered.empty:
            st.warning("Tidak ada data penjualan yang ditemukan untuk filter yang Anda pilih."); st.stop()
        
        # Judul Dashboard
        st.title(f"Dashboard Analisis Penjualan: {selected_branch}")
        st.markdown(f"Periode Analisis: **{start_date.strftime('%d %B %Y')}** hingga **{end_date.strftime('%d %B %Y')}**")

        # --- LAKUKAN SEMUA KALKULASI ANALISIS SEKALI SAJA ---
        with st.spinner("Menjalankan semua analisis..."):
            monthly_agg = analyze_monthly_trends(df_filtered)
            channel_results = calculate_channel_analysis(df_filtered)
            menu_results = calculate_menu_engineering(df_filtered)
            ops_results = calculate_operational_efficiency(df_filtered)
            price_group_results = calculate_price_group_analysis(df_filtered)
        
        # --- TAMPILKAN RINGKASAN EKSEKUTIF ---
        if monthly_agg is not None and len(monthly_agg) >= 3:
            summary = generate_executive_summary(monthly_agg, channel_results, menu_results, ops_results)
            display_executive_summary(summary)

        # --- TATA LETAK TAB ---
        trend_tab, ops_tab, strategic_tab = st.tabs([
            "📈 **Dashboard Tren Performa**", 
            "🚀 **Dashboard Analisis Operasional**",
            "🧠 **Analisis Strategis**"
        ])

        # --- KONTEN TAB 1: TREN PERFORMA ---
        with trend_tab:
            st.header("Analisis Tren Performa Jangka Panjang")
            if monthly_agg is not None and not monthly_agg.empty:
                display_monthly_kpis(monthly_agg)
                display_trend_chart_and_analysis(monthly_agg, 'TotalMonthlySales', 'Penjualan', 'royalblue')
                display_trend_chart_and_analysis(monthly_agg, 'TotalTransactions', 'Transaksi', 'orange')
                display_trend_chart_and_analysis(monthly_agg, 'AOV', 'AOV', 'green')
            else:
                st.warning("Tidak ada data bulanan yang cukup untuk analisis tren pada periode ini.")

        # --- KONTEN TAB 2: ANALISIS OPERASIONAL ---
        with ops_tab:
            st.header("Wawasan Operasional dan Taktis")
            display_channel_analysis(channel_results)
            st.markdown("---")
            display_menu_engineering(menu_results)
            st.markdown("---")
            display_operational_efficiency(ops_results)

        # --- KONTEN TAB 3: ANALISIS STRATEGIS ---
        with strategic_tab:
            st.header("Sintesis dan Analisis Silang")
            display_price_group_analysis(price_group_results)
            st.markdown("---")
            create_waiter_performance_analysis(df_filtered.copy())
            st.markdown("---")
            create_discount_effectiveness_analysis(df_filtered.copy())
            st.markdown("---")
            create_regional_analysis(df_filtered.copy())
# # ==============================================================================
# # LOGIKA AUTENTIKASI
# # ==============================================================================
# try:
#     config = {'credentials': st.secrets['credentials'].to_dict(), 'cookie': st.secrets['cookie'].to_dict()}
#     authenticator = stauth.Authenticate(
#         config['credentials'],
#         config['cookie']['name'],
#         config['cookie']['key'],
#         config['cookie']['expiry_days']
#     )
#     name, auth_status, username = authenticator.login("Login", "main")
#     if auth_status is False:
#         st.error("Username atau password salah.")
#     elif auth_status is None:
#         st.warning("Silakan masukkan username dan password.")
#     elif auth_status:
#         main_app(name)
# except KeyError as e:
#     st.error(f"❌ Kesalahan Konfigurasi 'secrets.toml': Key {e} tidak ditemukan.")
#     st.info("Pastikan file secrets Anda memiliki struktur [credentials] dan [cookie] yang benar sesuai dokumentasi streamlit-authenticator.")
# except Exception as e:
#     st.error(f"Terjadi kesalahan tak terduga saat inisialisasi: {e}")