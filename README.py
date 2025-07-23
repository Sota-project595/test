import pandas as pd
import plotly.graph_objects as go
import urllib.request
import json
import numpy as np

# ===== 1. データ読み込み =====

# 近年の巨大地震データをUSGSからダウンロード
url_earthquakes = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2000-01-01&minmagnitude=8"
try:
    with urllib.request.urlopen(url_earthquakes) as response:
        earthquake_data = json.load(response)
except Exception as e:
    print(f"地震データの取得に失敗しました: {e}")
    earthquake_data = None

# プレート境界のGeoJSONデータをダウンロード
url_boundaries = "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json"
with urllib.request.urlopen(url_boundaries) as response:
    plate_boundaries = json.load(response)

# 熱水噴出孔データの読み込み
try:
    file_path = 'vent_fields_all_20200325cleansorted.csv'
    df_vents = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')
except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりません。")
    exit()

# 必要な列の定義
lat_col = 'Latitude'
lon_col = 'Longitude'
name_col = 'Name.ID'
if not all(col in df_vents.columns for col in [lat_col, lon_col, name_col]):
    print("エラー: 必要な列（Latitude, Longitude, Name.ID）が見つかりません。")
    exit()


# ===== 2. 高エネルギー源の座標を準備 =====

def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- プレート境界に近い熱水噴出孔を抽出 ---
plate_nodes = []
for feature in plate_boundaries['features']:
    if feature['geometry']['type'] == 'LineString':
        plate_nodes.extend(feature['geometry']['coordinates'])

df_plate_nodes = pd.DataFrame(plate_nodes, columns=['lon', 'lat'])

min_distances = []
for index, vent in df_vents.iterrows():
    distances = haversine(vent[lon_col], vent[lat_col], df_plate_nodes['lon'], df_plate_nodes['lat'])
    min_distances.append(distances.min())

df_vents['min_dist_to_boundary'] = min_distances

threshold_km = 250
high_energy_vents = df_vents[df_vents['min_dist_to_boundary'] <= threshold_km]

# --- 巨大地震の座標を抽出 ---
eq_lons, eq_lats, eq_texts = [], [], []
if earthquake_data:
    for feature in earthquake_data['features']:
        eq_lons.append(feature['geometry']['coordinates'][0])
        eq_lats.append(feature['geometry']['coordinates'][1])
        mag = feature['properties']['mag']
        place = feature['properties']['place']
        eq_texts.append(f"M{mag} - {place}")

# --- 高エネルギー源となる全ポイントを統合 ---
he_points = list(zip(high_energy_vents[lon_col], high_energy_vents[lat_col])) + list(zip(eq_lons, eq_lats))
df_he_points = pd.DataFrame(he_points, columns=['lon', 'lat'])


# ===== 3. ヒートマップ用の密度データを計算 =====
print("高エネルギー領域の密度を計算中...（以前より時間がかかります）")

# (ここから変更) 地球全体を覆う格子の解像度を上げる
lon_grid_res = 360 # 経度の解像度 (120 -> 360)
lat_grid_res = 180 # 緯度の解像度 (60 -> 180)
# (ここまで変更)

lon_grid = np.linspace(-180, 180, lon_grid_res)
lat_grid = np.linspace(-90, 90, lat_grid_res)
z_grid = np.zeros((len(lat_grid), len(lon_grid)))

# 各エネルギー源が影響を及ぼす範囲（バンド幅）
bandwidth = 800.0

# 各グリッドポイントのエネルギー密度を計算
for i, lat in enumerate(lat_grid):
    for j, lon in enumerate(lon_grid):
        if not df_he_points.empty:
            distances = haversine(lon, lat, df_he_points['lon'], df_he_points['lat'])
            min_dist = distances.min()
            z_grid[i, j] = np.exp(-(min_dist**2) / (2 * bandwidth**2))

print("密度計算完了。")


# ===== 4. 地図の描画 =====

fig = go.Figure()

# --- プレート境界 ---
lons_b, lats_b = [], []
for feature in plate_boundaries['features']:
    geometry = feature['geometry']
    if geometry['type'] == 'LineString':
        coords = geometry['coordinates']
        lons_b.extend([coord[0] for coord in coords])
        lats_b.extend([coord[1] for coord in coords])
        lons_b.append(None)
        lats_b.append(None)

fig.add_trace(go.Scattergeo(
    lon=lons_b, lat=lats_b, mode='lines',
    line=dict(width=4, color='orange'),
    hoverinfo='none', name='プレート境界'
))

# --- 高エネルギー域をヒートマップとして描画 ---
grid_lons_flat = np.array([[lon for lon in lon_grid] for _ in lat_grid]).flatten()
grid_lats_flat = np.array([[lat for _ in lon_grid] for lat in lat_grid]).flatten()
grid_z_flat = z_grid.flatten()

threshold = 0.7
dense_lons = grid_lons_flat[grid_z_flat > threshold]
dense_lats = grid_lats_flat[grid_z_flat > threshold]
dense_z = grid_z_flat[grid_z_flat > threshold]

fig.add_trace(go.Scattergeo(
    lon=dense_lons,
    lat=dense_lats,
    mode='markers',
    marker=dict(
        color=dense_z,
        colorscale='Hot_r',
        showscale=True,
        colorbar=dict(title='エネルギー密度'),
        # (ここを変更) マーカーサイズを小さくして密度を高める
        size=3,
        opacity=0.5,
        line_width=0,
    ),
    hoverinfo='none',
    name='高エネルギー域'
))


# --- 熱水噴出孔のマーカー ---
fig.add_trace(go.Scattergeo(
    lon=df_vents[lon_col], lat=df_vents[lat_col], text=df_vents[name_col],
    hoverinfo='text', mode='markers',
    marker=dict(color='red', size=5, opacity=0.8, line=dict(width=1, color='black')),
    name='熱水噴出孔'
))

# --- 巨大地震のマーカー ---
if earthquake_data:
    fig.add_trace(go.Scattergeo(
        lon=eq_lons, lat=eq_lats, text=eq_texts,
        hoverinfo='text', mode='markers',
        marker=dict(color='yellow', size=15, symbol='star', line=dict(width=1, color='black')),
        name='地震：M8.0+'
    ))


# --- レイアウト設定 ---
fig.update_layout(
    title='今から、地球を動かす',
    geo=dict(
        projection_type='orthographic',
        showland=True, landcolor="rgb(224, 224, 224)",
        showocean=True, oceancolor="rgb(170, 221, 255)",
        showcoastlines=True, coastlinecolor="rgb(50, 50, 50)",
        showcountries=True, bgcolor='rgba(0,0,0,0)',
    ),
    paper_bgcolor='black',
    font=dict(color='white'),
    legend=dict(x=0.01, y=0.99, bordercolor='white', borderwidth=1),
    margin=dict(r=0, t=40, b=0, l=0),
    # (ここを追加) カーソルの反応範囲を広げる（ピクセル単位）
    hoverdistance=10
)

fig.show()
