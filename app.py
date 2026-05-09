"""
Spotify + Lyrics Explorer — Web App
=====================================
Artista principal: Fleetwood Mac
Basado en el notebook Spotify_Lyrics_Analysis.ipynb
"""

import os
import time
import re
import warnings
import ast
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA AUTOMÁTICA DEL DATASET DESDE GOOGLE DRIVE
# ID extraído de: https://drive.google.com/file/d/12ZjQa-A3Yddq_Fyh0w9ARvXrU8BYMvAt/view
# ─────────────────────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "12ZjQa-A3Yddq_Fyh0w9ARvXrU8BYMvAt"
DATA_FILE = "tracks_features.csv"


@st.cache_resource(show_spinner=False)
def download_dataset():
    """Descarga el CSV desde Google Drive si no existe ya en disco."""
    if os.path.exists(DATA_FILE):
        return True
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, DATA_FILE, quiet=False)
        return True
    except Exception as e:
        st.error(f"Error al descargar el dataset: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify + Lyrics — Fleetwood Mac",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ARTISTAS — solo álbumes de estudio (igual que el notebook)
# ─────────────────────────────────────────────────────────────────────────────
ARTISTS_CONFIG = {
    'Fleetwood Mac': {
        'spotify_id': '08GQAI4eElDnROBrJRGE0X',
        # Fuente: https://es.wikipedia.org/wiki/Fleetwood_Mac
        'studio_albums': {
            'Fleetwood Mac',
            'Mr. Wonderful',
            'Then Play On',
            'Kiln House',
            'Future Games',
            'Bare Trees',
            'Penguin',
            'Mystery to Me',
            'Heroes Are Hard to Find',
            'Rumours',
            'Tusk',
            'Mirage',
            'Tango in the Night',
            'Behind the Mask',
            'Time',
            'Say You Will',
        },
        'wiki': 'https://es.wikipedia.org/wiki/Fleetwood_Mac',
        'default_album': 'Behind the Mask',
        'default_song': 'Silver Springs',
    },
}

AUDIO_FEATURES = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'liveness', 'loudness', 'speechiness',
    'tempo', 'valence',
]
RADAR_FEATURES = ['acousticness', 'danceability', 'energy',
                  'instrumentalness', 'liveness', 'valence']

# Stop words igual que en el notebook
STOP_WORDS = {
    'i', 'me', 'my', 'you', 'your', 'the', 'a', 'an', 'and', 'or', 'but',
    'in', 'on', 'at', 'to', 'for', 'of', 'it', 'is', 'was', 'are', 'be',
    'have', 'had', 'do', 'did', 'will', 'would', 'can', 'could', 'that',
    'this', 'with', 'not', 'no', 'so', 'if', 'all', 'we', 'he', 'she',
    'they', 'them', 'their', 'its', 'what', 'when', 'where', 'how', 'why',
    'as', 'up', 'out', 'just', 'been', 'there', 'then', 'than', 'like',
    'from', 'into', 'about', 'get', 'got', 'oh', 'yeah', 'know', 'one',
    've', 're', 'll', 's', 't', 'm', 'dont', 'im', 'ive', 'aint', 'cause',
    'ain', 'em', 'gonna', 'wanna', 'gotta', 'let', 'come', 'go', 'say',
    'see', 'want', 'back', 'still', 'down', 'up', 'away', 'never', 'ever',
    'every', 'more', 'some', 'any', 'too', 'now', 'here', "it's",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — datos
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def artist_id_in_list(artist_ids_str: str, target_id: str) -> bool:
    try:
        return target_id in ast.literal_eval(str(artist_ids_str))
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def filter_artist(_df: pd.DataFrame, artist_name: str) -> pd.DataFrame:
    cfg = ARTISTS_CONFIG[artist_name]
    mask = _df['artist_ids'].apply(lambda x: artist_id_in_list(x, cfg['spotify_id']))
    adf = _df[mask].copy()
    # Normalizar nombre de álbum: quitar (Remastered), (Deluxe), etc.
    adf['short_album_name'] = adf['album'].str.split('(').str[0].str.strip()
    # Solo álbumes de estudio
    adf = adf[adf['short_album_name'].isin(cfg['studio_albums'])]
    # Deduplicar manteniendo el más antiguo
    adf = (
        adf.sort_values('year')
        .drop_duplicates(subset=['short_album_name', 'name'], keep='first')
        .reset_index(drop=True)
    )
    return adf


def get_album_order(adf: pd.DataFrame) -> list:
    return (
        adf.drop_duplicates('short_album_name')
        .sort_values('year')['short_album_name']
        .tolist()
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — letras (mismo fetch_lyrics / clean_lyrics / tokenize que el notebook)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_lyrics(artist: str, title: str) -> str | None:
    try:
        url = f"https://api.lyrics.ovh/v1/{requests.utils.quote(artist)}/{requests.utils.quote(title)}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json().get('lyrics') or None
        return None
    except Exception:
        return None


def clean_lyrics(text: str) -> str:
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def tokenize(text: str) -> list:
    # Igual que el notebook: palabras > 3 caracteres
    return [w for w in re.findall(r"[a-z']+", text.lower())
            if w not in STOP_WORDS and len(w) > 3]


def lyrics_stats(lyrics: str) -> dict:
    tokens = tokenize(lyrics)
    unique = set(tokens)
    return {
        'total_words': len(tokens),
        'unique_words': len(unique),
        'ttr': len(unique) / max(len(tokens), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS (misma lógica que el notebook)
# ─────────────────────────────────────────────────────────────────────────────
def plot_track_counts(adf: pd.DataFrame, album_order: list, artist_name: str):
    tc = adf.groupby('short_album_name')['name'].count().loc[album_order]
    colors = sns.color_palette('tab10', len(album_order))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(tc.index[::-1], tc.values[::-1], color=colors[::-1])
    ax.set_xlabel('Número de pistas')
    ax.set_title(f'Pistas por álbum de estudio — {artist_name}')
    plt.tight_layout()
    return fig


def plot_scatter(adf: pd.DataFrame, album_order: list, artist_name: str,
                 x_feat: str = 'valence', y_feat: str = 'acousticness'):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=adf, x=x_feat, y=y_feat,
        hue='short_album_name', hue_order=album_order,
        palette='tab10', size='duration_ms', sizes=(50, 800),
        alpha=0.75, ax=ax
    )
    handles, labels = ax.get_legend_handles_labels()
    n = len(album_order)
    ax.legend(handles[1:n+1], labels[1:n+1], title='Álbum',
              bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title(f'{x_feat} vs. {y_feat} — {artist_name}')
    plt.tight_layout()
    return fig


def plot_radar(adf: pd.DataFrame, album_order: list, artist_name: str):
    means = adf.groupby('short_album_name')[RADAR_FEATURES].mean().loc[album_order]
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
    N = len(RADAR_FEATURES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette('tab10', len(album_order))
    for (album, row), color in zip(norm.iterrows(), colors):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, color=color, linewidth=1.8, label=album)
        ax.fill(angles, vals, color=color, alpha=0.07)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_FEATURES, size=10)
    ax.set_title(f'Audio Features por álbum (normalizado) — {artist_name}',
                 size=12, pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.55, 1.15), fontsize=7)
    plt.tight_layout()
    return fig


def plot_boxplot(adf: pd.DataFrame, album_order: list, feat: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    order_present = [a for a in album_order if a in adf['short_album_name'].values]
    sns.boxplot(data=adf, x='short_album_name', y=feat,
                order=order_present, palette='tab10', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.set_title(f"Distribución de '{feat}' por álbum")
    ax.set_xlabel('')
    plt.tight_layout()
    return fig


def plot_wordcloud(lyrics_text: str, title: str):
    wc = WordCloud(
        width=900, height=450, background_color='black',
        colormap='plasma', stopwords=STOP_WORDS,
        max_words=100, collocations=False,
    ).generate(lyrics_text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_word_freq(freq: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette('plasma', len(freq))
    freq[::-1].plot(kind='barh', ax=ax, color=colors[::-1])
    ax.set_title(title)
    ax.set_xlabel('Frecuencia')
    plt.tight_layout()
    return fig


def plot_richness(vocab_data: dict, album_order: list, artist_name: str):
    albums = [a for a in album_order if a in vocab_data]
    unique_vals = [vocab_data[a]['unique_words'] for a in albums]
    ttr_vals = [vocab_data[a]['ttr'] for a in albums]
    colors = sns.color_palette('tab10', len(albums))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(albums[::-1], unique_vals[::-1], color=colors[::-1])
    axes[0].set_xlabel('Palabras únicas')
    axes[0].set_title(f'Vocabulario único por álbum\n{artist_name}')

    axes[1].barh(albums[::-1], ttr_vals[::-1], color=colors[::-1])
    axes[1].set_xlabel('TTR (Type-Token Ratio)')
    axes[1].set_title(f'Riqueza léxica (TTR) por álbum\n{artist_name}')

    plt.suptitle('Análisis de letras por álbum de estudio', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎸 Spotify + Lyrics")
    st.markdown("---")

    section = st.radio(
        "Sección",
        ["📁 Álbumes de Estudio", "📊 Audio Features", "📝 Análisis de Letras"],
    )

    st.markdown("---")
    # Solo un artista por ahora, fácil de ampliar
    artist_name = st.selectbox("Artista", list(ARTISTS_CONFIG.keys()))
    cfg = ARTISTS_CONFIG[artist_name]

    st.markdown("---")
    st.caption(
        f"Álbumes de estudio definidos según Wikipedia.\n\n"
        f"[Ver discografía ↗]({cfg['wiki']})"
    )

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA Y CARGA AUTOMÁTICA DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "adf" not in st.session_state:
    st.session_state.adf = None
if "current_artist" not in st.session_state:
    st.session_state.current_artist = None

if st.session_state.df is None:
    with st.spinner("⬇️ Descargando dataset desde Google Drive (solo la primera vez)..."):
        ok = download_dataset()
    if ok:
        with st.spinner("📂 Cargando dataset..."):
            try:
                st.session_state.df = load_dataset(DATA_FILE)
                st.sidebar.success(f"✅ Dataset listo: {len(st.session_state.df):,} filas")
            except Exception as e:
                st.sidebar.error(f"Error al leer el CSV: {e}")
    else:
        st.sidebar.error("No se pudo descargar el dataset.")

if st.session_state.df is not None and artist_name != st.session_state.current_artist:
    with st.spinner(f"Filtrando álbumes de estudio de {artist_name}..."):
        st.session_state.adf = filter_artist(st.session_state.df, artist_name)
        st.session_state.current_artist = artist_name

df = st.session_state.df
adf = st.session_state.adf

no_data = adf is None
if no_data:
    placeholder_msg = "⏳ Esperando a que se cargue el dataset..."

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — ÁLBUMES DE ESTUDIO (Punto 1)
# ─────────────────────────────────────────────────────────────────────────────
if section == "📁 Álbumes de Estudio":
    st.title(f"📁 Álbumes de Estudio — {artist_name}")
    st.markdown(
        f"""
        Los álbumes mostrados son únicamente los **álbumes de estudio**,
        excluyendo recopilaciones, directos, EPs y reediciones.
        Lista obtenida de [Wikipedia]({cfg['wiki']}).
        """
    )

    if no_data:
        st.info(placeholder_msg)
        st.stop()

    album_order = get_album_order(adf)

    col1, col2, col3 = st.columns(3)
    col1.metric("Álbumes de estudio", len(album_order))
    col2.metric("Pistas únicas", len(adf))
    col3.metric("Período", f"{int(adf['year'].min())}–{int(adf['year'].max())}")

    st.markdown("#### Resumen por álbum")
    summary = (
        adf.groupby('short_album_name')
        .agg(Pistas=('name', 'count'), Año=('year', 'min'))
        .loc[album_order]
    )
    st.dataframe(summary, use_container_width=True)

    st.markdown("#### Pistas por álbum")
    st.pyplot(plot_track_counts(adf, album_order, artist_name))

    st.markdown("#### Tabla de pistas")
    cols_show = ['name', 'short_album_name', 'year'] + [
        f for f in ['acousticness', 'valence', 'energy', 'danceability'] if f in adf.columns
    ]
    st.dataframe(
        adf[cols_show].rename(columns={
            'name': 'Pista', 'short_album_name': 'Álbum', 'year': 'Año'
        }),
        use_container_width=True, height=350
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — AUDIO FEATURES
# ─────────────────────────────────────────────────────────────────────────────
elif section == "📊 Audio Features":
    st.title(f"📊 Audio Features — {artist_name}")

    if no_data:
        st.info(placeholder_msg)
        st.stop()

    album_order = get_album_order(adf)
    feats_available = [f for f in AUDIO_FEATURES if f in adf.columns]

    tab_scatter, tab_radar, tab_box = st.tabs(
        ["Acousticness vs. Valence", "Radar Chart", "Distribuciones"]
    )

    with tab_scatter:
        st.markdown(
            "Relación entre **acousticness** y **valence** por álbum. "
            "El tamaño del punto representa la duración de la canción."
        )
        c1, c2 = st.columns(2)
        x_feat = c1.selectbox("Eje X", feats_available,
                               index=feats_available.index('valence') if 'valence' in feats_available else 0)
        y_feat = c2.selectbox("Eje Y", feats_available,
                               index=feats_available.index('acousticness') if 'acousticness' in feats_available else 1)
        st.pyplot(plot_scatter(adf, album_order, artist_name, x_feat, y_feat))
        st.caption(
            "Como se observa en la gráfica, no se detecta un gran número de canciones "
            "acústicas entre las pistas de los álbumes de estudio de Fleetwood Mac. "
            "El álbum *Heroes Are Hard to Find* destaca por tener mayor acousticness."
        )

    with tab_radar:
        st.markdown(
            "Características de audio promedio por álbum, normalizadas a [0, 1]. "
            "Permite comparar la 'firma sonora' de cada álbum."
        )
        st.pyplot(plot_radar(adf, album_order, artist_name))

    with tab_box:
        feat_box = st.selectbox("Feature a explorar", feats_available, key="boxfeat")
        st.pyplot(plot_boxplot(adf, album_order, feat_box))

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — ANÁLISIS DE LETRAS (Punto 2)
# ─────────────────────────────────────────────────────────────────────────────
elif section == "📝 Análisis de Letras":
    st.title(f"📝 Análisis de Letras — {artist_name}")
    st.markdown(
        "Letras obtenidas en tiempo real desde **[lyrics.ovh](https://lyrics.ovh/)**. "
        "Se analizan nube de palabras, frecuencia de vocabulario, riqueza léxica "
        "y se puede consultar la letra completa de cualquier canción."
    )

    if no_data:
        st.info(placeholder_msg)
        st.stop()

    album_order = get_album_order(adf)

    tab_wc, tab_song = st.tabs(
        ["☁️ Nube de palabras", "🎵 Letra de canción"]
    )

    # ── Nube de palabras ──────────────────────────────────────────────────
    with tab_wc:
        st.markdown("#### Nube de palabras por álbum")
        st.caption(
            "Se excluyen stop words (artículos, pronombres, verbos auxiliares) "
            "y palabras de ≤ 3 caracteres, igual que en el notebook."
        )

        chosen_album = st.selectbox("Álbum", album_order,
                                    index=album_order.index(cfg['default_album'])
                                    if cfg['default_album'] in album_order else 0)
        tracks = adf[adf['short_album_name'] == chosen_album]['name'].tolist()
        max_t = st.slider("Canciones a consultar", 3, min(12, len(tracks)),
                          min(8, len(tracks)))

        if st.button("🔍 Generar nube", key="wc_btn"):
            fetched = []
            prog = st.progress(0, text="Consultando lyrics.ovh…")
            for i, track in enumerate(tracks[:max_t]):
                lyr = fetch_lyrics(artist_name, track)
                if lyr:
                    fetched.append(clean_lyrics(lyr))
                prog.progress((i + 1) / max_t, text=f"({i+1}/{max_t}) {track}")
                time.sleep(0.4)
            prog.empty()

            if fetched:
                combined = ' '.join(fetched)
                st.success(f"✅ {len(fetched)}/{max_t} canciones con letra")

                st.pyplot(plot_wordcloud(
                    combined,
                    f'Nube de palabras — {chosen_album} ({artist_name})'
                ))

                tokens = tokenize(combined)
                freq = pd.Series(Counter(tokens)).sort_values(ascending=False).head(20)
                st.pyplot(plot_word_freq(freq, f'Top 20 palabras — {chosen_album}'))

                stats = lyrics_stats(combined)
                c1, c2, c3 = st.columns(3)
                c1.metric("Total palabras", f"{stats['total_words']:,}")
                c2.metric("Vocabulario único", f"{stats['unique_words']:,}")
                c3.metric("TTR", f"{stats['ttr']:.3f}")
            else:
                st.warning("No se encontraron letras. Prueba con otro álbum.")

    # ── Letra individual ──────────────────────────────────────────────────
    with tab_song:
        st.markdown("#### Ver letra de una canción")
        st.caption("Equivalente a la celda de *Silver Springs* del notebook.")

        chosen_album_s = st.selectbox("Álbum", album_order, key="song_alb")
        track_list = adf[adf['short_album_name'] == chosen_album_s]['name'].tolist()

        # Preseleccionar la canción por defecto del artista si está disponible
        default_idx = 0
        if cfg['default_song'] in track_list:
            default_idx = track_list.index(cfg['default_song'])
        chosen_track = st.selectbox("Canción", track_list,
                                    index=default_idx, key="song_track")

        if st.button("🎵 Obtener letra", key="song_btn"):
            with st.spinner("Consultando lyrics.ovh…"):
                lyr = fetch_lyrics(artist_name, chosen_track)

            if lyr:
                cleaned = clean_lyrics(lyr)
                stats = lyrics_stats(cleaned)

                c1, c2, c3 = st.columns(3)
                c1.metric("Total palabras", stats['total_words'])
                c2.metric("Vocabulario único", stats['unique_words'])
                c3.metric("TTR", f"{stats['ttr']:.3f}")

                with st.expander("📄 Letra completa", expanded=True):
                    st.text(cleaned)

                tokens = tokenize(cleaned)
                freq = pd.Series(Counter(tokens)).sort_values(ascending=False).head(15)
                if not freq.empty:
                    st.pyplot(plot_word_freq(
                        freq, f'Top 15 palabras — {chosen_track}'
                    ))
            else:
                st.warning(
                    f"No se encontró letra para **{chosen_track}** en lyrics.ovh. "
                    "Prueba con otra canción."
                )